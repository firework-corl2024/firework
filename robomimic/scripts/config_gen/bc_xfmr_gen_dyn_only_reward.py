from robomimic.scripts.config_gen.helper import *

task_names = [
    "PnPCounterToCab",
    "PnPCounterToSink",
    "TurnOffSinkFaucet",
    "OpenDoorSingleHinge",
    "TurnOnSinkFaucet",
    "CloseDoorSingleHinge",
]

task_names_mutex = [

    "rw2_put_the_blue_cup_in_the_basket",
    "rw2_put_the_red_bowl_in_the_basket",
    "rw4_pull_out_the_tray_of_the_oven_and_put_the_bowl_with_hot_dogs_on_the_tray",
    "rw4_put_the_book_in_the_back_compartment_of_the_caddy",
    "rw5_put_the_blue_mug_on_the_oven_tray",
    "rw5_put_the_bread_on_the_white_plate",
    "rw6_put_the_bowl_with_hot_dogs_on_the_white_plate",
    "rw6_put_the_pink_mug_on_the_white_plate",
    "rw7_put_the_book_in_the_back_compartment_of_the_caddy",
    "rw7_put_the_red_cup_in_the_front_compartment_of_the_caddy"

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
        #"""
        # EVAL_TASKS = ["PnPCounterToCab", "PnPCounterToStove", "TurnOnStove"]
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                (get_ds_cfg(task, src="round01", eval=EVAL_TASKS), task) for task in task_names
                ]
        )
        
        # different dynamics ckpt for spark or mutex
        generator.add_param(
            key="algo.dyn.load_ckpt",
            name="ckpt",
            group=235,
            values=[
            "/home/anonymous/expdata/spark/im/bc_xfmr/04-28-dyn_cvae/seed_123_ds_mg-100p_bs_8_no_dyn_debug_False_lr_0.0001_ld_no_action_True_determ_latent_False_use_embedding_loss_True_embedding_loss_w_1.0_dyn_train_embed_only_True_obs_keys_2images/20240428020249/models/model_epoch_400.pth",
            ],
            value_names=[
                "04-28-dyn_cvae_e400",
                ],
        )
        
    elif args.env == "mutex":
        EVAL_TASKS = None
        generator.add_param(
            key="train.data",
            name="ds",
            group=20000,
            values_and_names=[
                (get_ds_cfg(task, src="mutex", eval=EVAL_TASKS), task) for task in task_names_mutex
            ]
        )

        generator.add_param(
            key="train.data_format",
            name="",
            group=2,
            values=["mutex"],
        )
        
        # different dynamics ckpt for spark or mutex
        generator.add_param(
            key="algo.dyn.load_ckpt",
            name="ckpt",
            group=235,
            values=[
            "/home/anonymous/expdata/mutex/im/bc_xfmr/05-04-dyn_cvae/seed_123_ds_mutex_bs_8_no_dyn_debug_False_lr_0.0001_ld_no_action_True_determ_latent_False_use_embedding_loss_True_embedding_loss_w_1.0_dyn_train_embed_only_True_obs_keys_2images/20240504134807/models/model_epoch_400.pth",
            ],
            value_names=[
                "05-04-dyn_cvae_e400",
                ],
        )

    elif args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values_and_names=[
                # ([{"path": p} for p in scan_datasets("~/datasets/r2d2/success/2023-05-23_c2t-cans")], "pnp-c2t-cans"),
                # ([{"path": p} for p in scan_datasets("~/datasets/r2d2/success/2023-05-23_t2c-cans", postfix="trajectory_im84.h5")], "pnp-t2c-cans-84"),
                # ([{"path": p} for p in scan_datasets("~/datasets/r2d2/success/2023-05-23_t2c-cans", postfix="trajectory_im128.h5")], "pnp-t2c-cans-128"),
                # ([{"path": "~/tmp/Tue_May_23_12:11:04_2023/trajectory_im128.h5", "do_eval": False, "lang": "hello"}], "test"),
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

    generator.add_param(
        key="train.batch_size",
        name="bs",
        group=-1,
        values=[16],
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
        key="algo.dyn.train_reward",
        name="R",
        group=234,
        values=[True],
    )
    
    generator.add_param(
        key="experiment.validate",
        name="V",
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
    
    generator.add_param(
        key="train.classifier_weighted_sampling",
        name="ws",
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

    if True: # vae embed only

        generator.add_param(
            key="algo.dyn.deterministic",
            name="det",
            group=233,
            values=[False],
        )

        generator.add_param(
            key="algo.dyn.use_embedding_loss",
            name="",
            group=2334,
            values=[True],
        )

        generator.add_param(
            key="algo.dyn.embedding_loss_weight",
            name="",
            group=2334444,
            values=[1.0],
        )

        generator.add_param(
            key="algo.dyn.dyn_train_embed_only",
            name="",
            group=2334444,
            values=[True],
        )

    generator.add_param(
        key="algo.dyn.downscale_img",
        name="img84",
        group=234,
        values=[True], # False for the 04-11 version
    )

    # For reward classifier training
    
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

    generator.add_param(
        key="algo.optim_params.policy.learning_rate.scheduler_type",
        name="",
        group=234,
        values=["constant"], # no warmup
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    assert args.train_reward
    
    make_generator(args, make_generator_helper)
