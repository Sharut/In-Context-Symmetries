# https://github.com/facebookresearch/SIE/blob/main/src/dataset.py
from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

class Dataset3DIEBenchRotColor(Dataset):
    def __init__(self, dataset_root, img_file, labels_file,experience="quat", size_dataset=-1, transform=None, mode='pretraining', args=None):
        self.dataset_root = dataset_root
        self.mode = mode
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience  
        self.args = args
        self.cache_x, self.cache_r, self.cache_y = self._create_cache()  
        self.cache = (self.cache_x, self.cache_r, self.cache_y)
        print('Cache created')

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img
    
    def _create_cache(self):
        generator = torch.Generator().manual_seed(0)
        idxs = torch.randint(0, len(self.samples), (self.args.block_size//2,), generator=generator)
        view_idxs = [np.random.choice(50, 2, replace=False) for _ in idxs]
        view_1 = [self.get_img(os.path.join(self.dataset_root, self.samples[idx][1:], f"image_{view_idxs[c][0]}.jpg")) for c, idx in enumerate(idxs)]
        view_2 = [self.get_img(os.path.join(self.dataset_root, self.samples[idx][1:], f"image_{view_idxs[c][1]}.jpg")) for c, idx in enumerate(idxs)]
        
        latent_1 = [np.load(os.path.join(self.dataset_root, self.samples[idx][1:], f"latent_{view_idxs[c][0]}.npy")).astype(np.float32) for c, idx in enumerate(idxs)]
        latent_2 = [np.load(os.path.join(self.dataset_root, self.samples[idx][1:], f"latent_{view_idxs[c][1]}.npy")).astype(np.float32) for c, idx in enumerate(idxs)]

        angles_1 = [latent_1[c][:3] for c in range(len(latent_1))]
        angles_2 = [latent_2[c][:3] for c in range(len(latent_2))]
        rot_1 = [R.from_euler("xyz",angles_1[c]) for c in range(len(angles_1))]
        rot_2 = [R.from_euler("xyz",angles_2[c]) for c in range(len(angles_2))]
        rot_1_to_2 = [rot_1[c].inv()*rot_2[c] for c in range(len(rot_1))]
        if self.experience == "quat":
            angles = [rot_1_to_2[c].as_quat().astype(np.float32) for c in range(len(rot_1_to_2))]
            rot_1 = [rot_1[c].as_quat().astype(np.float32) for c in range(len(rot_1))]
            rot_2 = [rot_2[c].as_quat().astype(np.float32) for c in range(len(rot_2))]
        else:
            angles = [rot_1_to_2[c].as_euler("xyz").astype(np.float32) for c in range(len(rot_1_to_2))]
        
        other_params = [latent_2[c][[3,6]] - latent_1[c][[3,6]] for c in range(len(latent_1))]
        latent_total = [np.concatenate((angles[c],other_params[c])) for c in range(len(angles))]
        individual_params = [np.stack([rot_1[c],rot_2[c]], axis=0) for c in range(len(rot_1))]
        other_params = [np.stack([latent_1[c][[3,6]],latent_2[c][[3,6]]], axis=0) for c in range(len(latent_1))]
        
        x1, x2 = torch.stack(view_1), torch.stack(view_2)
        targets = torch.tensor([self.labels[idx] for idx in idxs])
        y = torch.stack([targets,targets], dim=1).view(-1).cuda(non_blocking=True)
        sizes = x1.size()
        x = torch.stack((x1, x2), dim=1).view(sizes[0]*2, sizes[1], sizes[2], sizes[3]).float()

        return x, torch.FloatTensor(latent_total), y

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        img_1 = self.get_img(os.path.join(self.dataset_root, self.samples[i][1:], f"image_{views[0]}.jpg"))
        img_2 = self.get_img(os.path.join(self.dataset_root, self.samples[i][1:], f"image_{views[1]}.jpg"))        
    
        latent_1 =np.load(os.path.join(self.dataset_root, self.samples[i][1:], f"latent_{views[0]}.npy")).astype(np.float32)
        latent_2 =np.load(os.path.join(self.dataset_root, self.samples[i][1:], f"latent_{views[1]}.npy")).astype(np.float32)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
            rot_1 = rot_1.as_quat().astype(np.float32)
            rot_2 = rot_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)
        
        # relative params
        other_params = latent_2[[3,6]] - latent_1[[3,6]]
        latent_total = np.concatenate((angles,other_params))
        
        # individual params
        latent_v1 = np.concatenate((rot_1,latent_1[[3,6]]))
        latent_v2 = np.concatenate((rot_2,latent_2[[3,6]]))
        individual_params = np.stack([latent_v1,latent_v2], axis=0)

        return img_1, img_2, torch.FloatTensor(latent_total), individual_params, label

    def __len__(self):
        return len(self.samples)


class EvalDatasetAll(Dataset):
    def __init__(self, dataset_root, img_file, labels_file, experience="quat", size_dataset=-1, transform=None, args=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience  
        self.args = args

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img
    
    def __getitem__(self, i):
        labelidx = i // 50
        viewidx = i % 50
        label = self.labels[labelidx]
        # Latent vector creation
        # views = np.random.choice(50,1, replace=False)
        img = self.get_img(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"image_{viewidx}.jpg"))    
        latent = np.load(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"latent_{viewidx}.npy")).astype(np.float32)
        
        # change rotation angle to quat mode
        r_angle = R.from_euler("xyz",latent[:3])
        latent = np.concatenate((r_angle.as_quat().astype(np.float32), latent[[3,6]]))
        return img, torch.FloatTensor(latent), label

    def __len__(self):
        return len(self.samples)*50
    



class MRREvalDatasetAll(Dataset):
    def __init__(self, dataset_root, img_file, labels_file, experience="quat", size_dataset=-1, transform=None, args=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience  
        self.args = args

    def get_img(self, path, is_transform=True):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform and is_transform:
                img = self.transform(img) 
        return img
    
    def _get_img_from_index(self, labelidx, local_end):
        img_1 = self.get_img(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"image_{local_end}.jpg"), is_transform=False)   
        latent_1 = np.load(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"latent_{local_end}.npy")).astype(np.float32)
        return img_1, latent_1

    def __getitem__(self, i):
        labelidx = i // 50
        # labelidx = 2990
        start = 0
        end = i % 50
        label = self.labels[labelidx]
        
        img_1 = self.get_img(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"image_{start}.jpg"))   
        latent_1 = np.load(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"latent_{start}.npy")).astype(np.float32)
        img_2 = self.get_img(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"image_{end}.jpg"))    
        latent_2 = np.load(os.path.join(self.dataset_root, self.samples[labelidx][1:], f"latent_{end}.npy")).astype(np.float32)
        

        rot_1 = R.from_euler("xyz",latent_1[:3])
        rot_2 = R.from_euler("xyz",latent_2[:3])
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)
        
        # relative params
        other_params = latent_2[[3,6]] - latent_1[[3,6]]
        latent_total = np.concatenate((angles,other_params))

        global_start = i // 50 * 50
        global_end = global_start + end
        assert global_end==i
        return img_1, img_2, torch.FloatTensor(latent_total), global_start, global_end, labelidx, start, end

    def __len__(self):
        return len(self.samples)*50