from robomimic.scripts.config_gen.helper import *
import re

task_names = [
    "CloseDoorSingleHinge",
    "OpenDoorDoubleHinge",
    "OpenDoorSingleHinge",
    "TurnOnSinkFaucet",
    "TurnOnMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffMicrowave",
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "CoffeeSetupMug",
    "CoffeeServeMug"
]

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer_dyn_only.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    if args.env == "spark":
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                (get_ds_cfg("CloseDoorSingleHinge", src="human", eval=EVAL_TASKS), "human"), 
                (get_ds_cfg("TurnOnSinkFaucet", src="human", eval=EVAL_TASKS), "human"),
                (get_ds_cfg("TurnOnMicrowave", src="human", eval=EVAL_TASKS), "human"),
                (get_ds_cfg("PnPCabToCounter", src="human", eval=EVAL_TASKS), "human"),
                (get_ds_cfg("CoffeeSetupMug", src="human", eval=EVAL_TASKS), "human"),
                # (get_ds_cfg("single_stage", src="mg", eval=EVAL_TASKS, filter_frac=1.0), "mg-100p"),
            ]
        )

        if args.train_reward:

            generator.add_param(
                key="train.data",
                name="ds",
                group=2,
                values_and_names=[
                    (get_ds_cfg(task, src="round01", eval=EVAL_TASKS), task) for task in task_names
                ]
            )


        generator.add_param(
            key="algo.dyn.obs_sequence",
            name="obs_keys",
            group=23345,
            values=[
                   ["robot0_agentview_left_image",
                    "robot0_eye_in_hand_image",]
                ],
            value_names=["2images"]
        )

    elif args.env == "mutex":
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                (get_ds_cfg("mutex", src="mutex", eval=EVAL_TASKS), "mutex"),
            ]
        )

        generator.add_param(
            key="algo.dyn.obs_sequence",
            name="obs_keys",
            group=23345,
            values=[
                       ["agentview_rgb",
                        "eye_in_hand_rgb"]
                ],
            value_names=["2images"]
        )

        generator.add_param(
            key="train.data_format",
            name="",
            group=2,
            values=["mutex"],
        )

    elif args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                ([{"path": p, "do_eval": False, "lang": "pick the can from the counter and place it in the sink"
                    } for p in scan_datasets("~/datasets/r2d2/success/2023-05-29-t2s-cans", postfix="trajectory_im128.h5")], "pnp-counter-to-sink-can"),
            ],
        )


    # train for 2k epochs and save every 100 epochs to be safe
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2000],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[100],
    )

    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[6],
    )

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    generator.add_param(
        key="train.seq_length",
        name="",
        group=-1,
        values=[20],
    )

    generator.add_param(
        key="train.frame_stack",
        name="",
        group=-1,
        values=[20],
    )

    generator.add_param(
        key="algo.transformer.context_length",
        name="",
        group=-1,
        values=[20],
    )
    
    generator.add_param(
        key="train.batch_size",
        name="bs",
        group=-1,
        values=[8],
    )
    
    generator.add_param(
        key="algo.dyn.no_dyn_debug",
        name="no_dyn_debug",
        group=233,
        values=[False],
    )
    
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial",
        name="lr",
        group=234,
        values=[1e-4],
    )

    generator.add_param(
        key="observation.modalities.obs.low_dim",
        name="ld",
        group=236,
        values=[
            []
            ],
    )

    generator.add_param(
        key="algo.dyn.no_action",
        name="no_action",
        group=233,
        values=[True],
    )

    if args.dyn_mode == "deterministic":
        
        generator.add_param(
            key="algo.dyn.deterministic",
            name="determ_latent",
            group=233,
            values=[True],
        )
        
    elif args.dyn_mode == "vae_embed_only":

        generator.add_param(
            key="algo.dyn.deterministic",
            name="determ_latent",
            group=233,
            values=[False],
        )

        generator.add_param(
            key="algo.dyn.use_embedding_loss",
            name="use_embedding_loss",
            group=2334,
            values=[True],
        )

        generator.add_param(
            key="algo.dyn.embedding_loss_weight",
            name="embedding_loss_w",
            group=2334444, 
            values=[1.0],
        )

        generator.add_param(
            key="algo.dyn.dyn_train_embed_only",
            name="dyn_train_embed_only",
            group=2334444,
            values=[True],
        )

    elif args.dyn_mode == "vae_image":

        generator.add_param(
            key="algo.dyn.deterministic",
            name="determ_latent",
            group=233,
            values=[False],
        )

        generator.add_param(
            key="algo.dyn.dyn_cell_type",
            name="dyn_cell_type",
            group=2334,
            values=["no_fusion"],
        )

        generator.add_param(
            key="algo.dyn.use_embedding_loss",
            name="use_embedding_loss",
            group=23344,
            values=[True],
        )

        generator.add_param(
            key="algo.dyn.embedding_loss_weight",
            name="embedding_loss_w",
            group=2334444,
            values=[0.01],
        )

        generator.add_param(
            key="algo.dyn.recons_full_batch",
            name="recons_full_batch",
            group=23344,
            values=[False],
        )

    if args.train_reward:
        
        generator.add_param(
            key="algo.dyn.train_reward",
            name="train_reward",
            group=234,
            values=[True],
            value_names=["T"],
        )
        
        """ Validate = True for Reward """
        
        generator.add_param(
            key="experiment.validate",
            name="",
            group=234,
            values=[True],
        )
        
        generator.add_param(
            key="train.hdf5_filter_key",
            name="",
            group=234,
            values=["train"],
        )
        
        generator.add_param(
            key="train.hdf5_validation_filter_key",
            name="",
            group=234,
            values=["valid"],
        )
        
        """ Need weighted sampling """
        
        generator.add_param(
            key="train.classifier_weighted_sampling",
            name="",
            group=234,
            values=[True],
        )    
        
        generator.add_param(
            key="train.dataset_keys",
            name="",
            group=234,
            values=[
                ["actions",
                "intv_labels"
                ]
                ],
        )
        
        generator.add_param(
            key="train.frame_stack",
            name="",
            group=-1,
            values=[1],
        )

        generator.add_param(
            key="train.pad_frame_stack",
            name="",
            group=-1,
            values=[False],
        )

        """ Training parameters """

        generator.add_param(
            key="train.batch_size",
            name="bs",
            group=-1,
            values=[8],
        )

        generator.add_param(
            key="algo.optim_params.policy.learning_rate.scheduler_type",
            name="",
            group=234,
            values=["constant"], # no warmup
        )
        
        """ Different saving for reward training """
        
        generator.add_param(
            key="experiment.save.every_n_epochs",
            name="",
            group=234,
            values=[20], 
        )

        generator.add_param(
            key="train.num_epochs",
            name="",
            group=234,
            values=[41], 
        )

        """ Load checkpoint for reward classifier """

        assert args.reward_ckpt is not None

        generator.add_param(
            key="algo.dyn.load_ckpt",
            name="ckpt",
            group=235,
            values=[
            args.reward_ckpt
            ],
            value_names=[
                format_ckpt_path(args.reward_ckpt)
                ],
        )

    return generator

def format_ckpt_path(path):
    # Split the path into parts
    parts = path.split('/')
    folder_name = parts[-5]

    epoch_match = re.search(r'model_epoch_(\d+).pth', parts[-1])
    epoch_number = epoch_match.group(1)
    
    formatted_string = f"{folder_name}_e{epoch_number}"
    
    return formatted_string

if __name__ == "__main__":
    parser = get_argparser()

    parser.add_argument(
        "--dyn_mode",
        type=str,
        choices=['deterministic',
                 'vae_embed_only',
                 'vae_image',
                 ],
        default='vae_embed_only'
    )         


    """
    parser.add_argument(
        "--train_reward",
        action="store_true",
    )  
    """

    parser.add_argument(
        "--reward_ckpt",
        default=None,
    )         

    args = parser.parse_args()
    
    make_generator(args, make_generator_helper)
