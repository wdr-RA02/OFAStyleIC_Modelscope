import argparse
from utils.config_modify_fn import cfg_modify_fn

if __name__=="__main__":
    parser=argparse.ArgumentParser(prog="model_operator",description="OFA Style model operator")
    parser.add_argument("mode", choices=["train", "eval", "inference"], help="what you want to do")
    parser.add_argument("-c", "--conf", help="trainer config json", type=str, required=True)
    parser.add_argument("-t", "--checkpoint", help="checkpoint", type=str)
    parser.add_argument("-e", "--max_epoches", help="max epoch", type=int, default=3)
    parser.add_argument("-b", "--batch_size", help="#samples per batch", type=int, default=4)
    parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)
    parser.add_argument("-p","--patch_image_size", help="patch image size in img embedding, default to 224",type=int, default=224)
    parser.add_argument("-m","--max_image_size", help="patch image size in img embedding, default to 256",type=int, default=256)
    args=parser.parse_args()

    mod_fn=cfg_modify_fn(args.max_epoches, 
                        args.batch_size, 
                        args.num_workers, 
                        patch_img_size=args.patch_image_size, 
                        max_img_size=args.max_image_size)
    if args.mode=="train":
        from model.finetuner import train
        train(args, mod_fn)
    
    elif args.mode=="eval":
        from model.evaluator import evaluate
        evaluate(args, mod_fn)

    elif args.mode=="inference":
        from model.inference_pipeline import inference
        inference(args, mod_fn)
