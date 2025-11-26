import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel import GlocalTextPathNavCMT
    
    model_name_or_path = args.model.pretrained_ckpt
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, weights_only=True)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v

    vis_config = PretrainedConfig.from_pretrained('bert-base-uncased')
    
    vis_config.gmap_max_action_steps = args.model.gmap_max_action_steps
    vis_config.image_feat_size = args.feature.image_feat_size
    vis_config.angle_feat_size = args.feature.angle_feat_size
    vis_config.obj_feat_size = args.feature.obj_feat_size
    vis_config.enable_og = args.feature.enable_og
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.model.num_l_layers
    vis_config.num_pano_layers = args.model.num_pano_layers
    vis_config.num_x_layers = args.model.num_x_layers
    vis_config.graph_sprels = args.model.graph_sprels
    vis_config.glocal_fuse = args.agent.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.model.fix_lang_embedding
    vis_config.fix_pano_embedding = args.model.fix_pano_embedding
    vis_config.fix_local_branch = args.model.fix_local_branch

    vis_config.update_lang_bert = not args.model.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.use_lang2visn_attn = False
    vis_config.hidden_dropout_prob = args.model.hidden_dropout_prob
    vis_config.attention_probs_dropout_prob = args.model.attn_dropout_prob

    vis_config.load_text_features = args.model.load_text_features
    vis_config.text_feat_size = args.model.text_feat_size

    vis_config.use_moe_layer = args.model.use_moe_layer
    vis_config.moe_type = args.model.moe_type
    vis_config.moe_position = args.model.moe_position
    vis_config.task_routing_feature = args.model.task_routing_feature
    vis_config.num_tasks = 20
    vis_config.num_experts = args.model.num_experts
    vis_config.num_experts_per_tok = args.model.num_experts_per_tok
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
