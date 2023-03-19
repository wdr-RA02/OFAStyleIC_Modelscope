import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser(prog="model_operator",description="OFA Style model operator")
    parser.add_argument("mode", choices=["train", "eval", "inference"], help="what you want to do")
    parser.add_argument("-c", "--conf", help="trainer config json", type=str, required=True)
    parser.add_argument("-p", "--checkpoint", help="checkpoint", type=str)
    parser.add_argument("-e", "--max_epoches", help="max epoch", type=int, default=3)
    parser.add_argument("-b", "--batch_size", help="#samples per batch", type=int, default=4)
    parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)
    args=parser.parse_args()

    if args.mode=="train":
        from model.finetuner import train
        train(args)
    
    elif args.mode=="eval":
        from model.evaluator import evaluate
        evaluate(args)

    elif args.mode=="inference":
        from model.inference_pipeline import inference
        inference(args)
