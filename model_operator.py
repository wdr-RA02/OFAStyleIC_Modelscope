import argparse

def train_fn(args):
    global parser
    if args.cider and args.checkpoint is None:
        parser.error("a checkpoint must be specified when using CIDEr finetuning")
    
    if not args.itm and args.itm_alpha:
        parser.error("--itm_alpha argument is available only when --itm is specified")
    
    global mod_fn
    from model.finetuner import train
    train(args, mod_fn, use_cider_scst=args.cider)

def eval_fn(args):
    global mod_fn
    from model.evaluator import evaluate
    evaluate(args, mod_fn)

def inference_fn(args):
    global mod_fn
    from model.inference_pipeline import inference
    inference(args, mod_fn)

def add_common_args(subparser):
    # add commom parsers
    subparser.add_argument("-c", "--conf", help="trainer config json", type=str, required=True)
    subparser.add_argument("-b", "--batch_size", help="#samples per batch", type=int, default=4)
    subparser.add_argument("-p","--patch_image_size", help="patch image size in img embedding, default to 224",type=int, default=224)
    subparser.add_argument("-m","--max_image_size", help="patch image size in img embedding, default to 256",type=int, default=256)


if __name__=="__main__":
    parser=argparse.ArgumentParser(prog="model_operator",description="OFA Style model operator")
    sub_parser=parser.add_subparsers(help="describes the user's expected action") 
    
    train_parser=sub_parser.add_parser("train", help="loads trainer")
    train_parser.set_defaults(callback=train_fn)
    
    add_common_args(train_parser)
    train_parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)
    train_parser.add_argument("-t", "--checkpoint", help="checkpoint", type=str)
    train_parser.add_argument("-e", "--max_epoches", help="max epoch", type=int, default=3)
    train_parser.add_argument("--lr", help="Adam learning rate, defaults to 5e-5", type=float)
    train_parser.add_argument("--lr_end", help="lr_scheduler end, defaults to 1e-7", type=float)
    train_parser.add_argument("--warm_up", help="Adam warm-up rate, defaults to 0.01", type=float)
    train_parser.add_argument("--weight_decay", help="weight decay, defaults to 0.001", type=float)
    train_parser.add_argument("--max_len", help="max length for beam search, defaults to 16", type=int)
    train_parser.add_argument("--beam_size", help="beam size for beam search, defaults to 5", type=int)
    train_parser.add_argument("--freeze_resnet", help="freeze resnet during training", action="store_true")
    # we limit that cider and itm cannot coexist
    train_tasks=train_parser.add_mutually_exclusive_group()
    train_tasks.add_argument("--cider", help="Execute CIDEr SCST finetune (experimental)", action="store_true")
    train_tasks.add_argument("--itm", help="Perform Image-text Matching pretrain task", action="store_true")
    train_parser.add_argument("--itm_alpha", help="Alpha weight for ITM task loss", default=1.0, type=float)


    eval_parser=sub_parser.add_parser("eval", help="loads evaluator")
    eval_parser.set_defaults(callback=eval_fn)
    eval_parser.add_argument("-l","--log_csv_file",help="where to save the csv file with eval results", type=str)
    add_common_args(eval_parser)
    eval_parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)

    inference_parser=sub_parser.add_parser("inference", help="loads inference_pipeline")
    inference_parser.set_defaults(callback=inference_fn)
    add_common_args(inference_parser)
    # -r and -j will be conflict
    infr_group=inference_parser.add_mutually_exclusive_group(required=True)
    infr_group.add_argument("-r", "--random", help="Randomly select samples from eval set",action="store_true")
    infr_group.add_argument("-j", "--inference_json", help="Extract items to be inferenced from json file", type=str)
    inference_parser.add_argument("-d", "--description", help="Description that will be written in result json \
                                  rather than base model name", type=str)
    inference_parser.add_argument("-o", "--output_dir", help="specify the place to output inference result json \
                                  rather than \{work_dir\}",type=str)
    
    args=parser.parse_args()

    from utils import cfg_modify_fn, reg_module
    mod_fn=cfg_modify_fn(args)

    args.callback(args)