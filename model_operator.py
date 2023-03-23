import argparse
from utils.train_conf import load_train_conf

def train_fn(args):
    global mod_fn
    from model.finetuner import train
    train(args, mod_fn)

def eval_fn(args):
    global mod_fn
    from model.evaluator import evaluate
    evaluate(args, mod_fn)

def inference_fn(args):
    from model.inference_pipeline import inference
    inference(args)

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

    eval_parser=sub_parser.add_parser("eval", help="loads evaluator")
    eval_parser.set_defaults(callback=eval_fn)
    add_common_args(eval_parser)

    eval_parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)
    inference_parser=sub_parser.add_parser("inference", help="loads inference_pipeline")
    inference_parser.set_defaults(callback=inference_fn)
    add_common_args(inference_parser)

    args=parser.parse_args()

    from model.utils.config_modify_fn import cfg_modify_fn
    from utils import reg_module
    
    mod_fn=cfg_modify_fn(args.max_epoches, 
                        args.batch_size, 
                        args.num_workers, 
                        patch_img_size=args.patch_image_size, 
                        max_img_size=args.max_image_size,
                        prompt=load_train_conf(args.conf).get("prompt", None))

    args.callback(args)