import torch
import torch.nn as nn
from gptmodel import GPT, GPTConfig


# We define a separate class to have Adam optimizer for all online networks such as rotation predictor, equivariance predictor, etc.
class OnlinePredictor(nn.Module):
    def __init__(self, feature_dim, predictor_dim, n_classes=10, args=None):
        super(OnlinePredictor, self).__init__()
        self.predictor_dim = predictor_dim
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.args = args

        # Online linear probe
        self.online_linear_probe_feat = torch.nn.Linear(feature_dim, n_classes).cuda()
        self.online_linear_probe_pred = torch.nn.Linear(self.predictor_dim, n_classes).cuda()

    def forward(self, feature, prediction, sup=False):
        class_logits_feat = self.online_linear_probe_feat(feature.detach())
        if sup:
            class_logits_pred = self.online_linear_probe_pred(prediction)
        else:
            class_logits_pred = self.online_linear_probe_pred(prediction.detach())
        return class_logits_feat, class_logits_pred

class BranchNetwork(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, args=None):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features
        self.eval_predictor = args.eval_predictor
        self.projection_dim = projection_dim
        r_dim = 6
        
        # Online network
        if args.equiv_eval == 'linear':
            self.equiv_linear = nn.Linear(self.projection_dim, r_dim)
        elif args.equiv_eval == 'mlp':
            self.equiv_linear = nn.Sequential(
                nn.Linear(self.projection_dim,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, r_dim),
            )

        # Backbone network (encoder, projector and predictor)
        self.enc.fc = nn.Identity()  # remove final fully connected layer.
        self.projector = nn.Identity()
        config = GPTConfig(args)
        self.predictor = GPT(config)
            
        # Rotation predictor
        self.aug_predictor = nn.Sequential(
                                    nn.Linear(self.projection_dim, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, args.augmentation_dim),
                                )
        
        self.rel_aug_predictor = nn.Sequential(nn.Linear(projection_dim*2,1024),
                                               nn.ReLU(),
                                               nn.Linear(1024,1024),
                                               nn.ReLU(),
                                               nn.Linear(1024, args.augmentation_dim),
                                            )
        

    def forward(self, x, r, past=None, eval_only=False, return_cache=False, cache_inference_mode = False, random_mask=False, mask_prob=0.15):
        feature = self.enc(x)
        ema_projection = None
        if not eval_only:
            projection = self.projector(feature)
            prediction, kv_cache = self.predictor(projection, r, past, cache_inference_mode, random_mask=random_mask, mask_prob=mask_prob)
        else:
            projection = prediction = None

        if return_cache:
            return feature, projection, prediction, ema_projection, kv_cache
        return feature, projection, prediction, ema_projection
    
    def forward_eval_train(self, x, r_dict, past_dict=None):
        # past is a dict {mode: {test_at: initial_past}}
        # r is a dict: {mode: r}
        # output_dict: {test_mode: [feature, {ctxt_len: prediction_zr}]}
        feature = self.enc(x)
        output_dict = {}
        for mode, r in r_dict.items():
            prediction_dict_zr = {}
            if past_dict is not None:
                for ctxt_len in past_dict[mode].keys():
                    prediction_zr, kv_cache = self.predictor(feature, r, past_dict[mode][ctxt_len], inference=True, random_mask=False)
                    prediction_dict_zr[ctxt_len] = prediction_zr
                output_dict[mode] = [feature, prediction_dict_zr]#, prediction_dict_zplus]
            else:
                prediction_zr, kv_cache = self.predictor(feature, r, past_dict, inference=False, random_mask=False)                
                output_dict[mode] = [feature, prediction_zr]#, prediction_zplus]
        return output_dict
    

    def forward_eval_rplus(self, x, r, past_dict):
        feature = self.enc(x)
        prediction_dict_zplus = {}
        for ctxt_len in past_dict.keys():
            prediction_zplus, kv_cache = self.predictor(feature, r, past_dict[ctxt_len], inference=True, random_mask=False)
            prediction_dict_zplus[ctxt_len] = prediction_zplus
        return feature, prediction_dict_zplus

