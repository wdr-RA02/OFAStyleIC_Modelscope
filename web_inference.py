import argparse
import os
import gradio as gr

from typing import Dict, Callable
from modelscope.pipelines import pipeline
from model.utils.constants import (Tasks, load_train_conf, \
                        generate_style_dict, cfg_modify_fn, OutputKeys)
from model.preprocessor.stylish_image_caption import OfaPreprocessorforStylishIC

def generate_infr_pipeline(train_conf: Dict[str, str], 
                           style_dict: Dict[str, int],
                           mod_fn: Callable):
    model_dir=os.path.join(train_conf["work_dir"], "output")
    tokenize=train_conf["tokenize_style"]
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir, cfg_modify_fn=mod_fn)
    if tokenize:
        assert isinstance(style_dict, dict)
        print("Tokenize style=True, add style dict. ")
        preprocessor.preprocess.add_style_token(style_dict)

    return pipeline(Tasks.image_captioning, 
                        model=model_dir, 
                        preprocessor=preprocessor)


def build_real_args(args):
    '''
    build args for model module from input
    '''
    int_parser=argparse.ArgumentParser()
    int_parser.add_argument("--conf", type=str)
    int_parser.add_argument("-m",dest="max_image_size", type=int)
    int_parser.add_argument("-p",dest="patch_image_size", type=int)
    int_args=int_parser.parse_args(["--conf", args.conf,
                             "-m", str(args.image_res), 
                             "-p", str(args.image_res)]) 
    
    return int_args


def inference(img_dir:str, style: str, res: int) -> str:
    global out_args
    out_args.image_res=res
    args=build_real_args(out_args)
    # load train config and style dict
    train_conf=load_train_conf(args.conf)
    style_dict=generate_style_dict(train_conf)
    mod_fn=cfg_modify_fn(args)

    model=generate_infr_pipeline(train_conf, style_dict=style_dict, mod_fn=mod_fn)
    assert os.path.exists(img_dir) and (style in style_dict.keys())

    input_dict={"image": img_dir, "style": style}
    result=model(input_dict).get(OutputKeys.CAPTION)

    return result[0]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("-p", "--port", type=int, default=8080)
    # merge patch and max
    parser.add_argument("-r", "--image_res", type=int, default=224)
    out_args=parser.parse_args()

    train_conf=load_train_conf(out_args.conf)
    style_dict=generate_style_dict(train_conf)


    with gr.Blocks(title="OFAStyle") as demo:
        gr.Markdown(f'''
        # OFAStyle
        ## Using config file: {out_args.conf} 
        ## Base model: {train_conf["model_name"]}, {train_conf["model_revision"]}
        ''')
        img=gr.Image(type="filepath", label="Image")
        style=gr.Dropdown(choices=list(style_dict.keys()), label="Personality", 
                        info="Pick a personality for the caption of this image")
        res=gr.Slider(128, 384,
                      value=out_args.image_res,
                      interactive=True,
                      step=1.0, label="Image resolution", 
                      info="The resolution that will be fed into model")

        cap=gr.Textbox(label="Generated caption", placeholder="Output will be shown here")
        btn=gr.Button("Generate Caption")
        btn.click(fn=inference, 
                inputs=[img, style, res], outputs=[cap])

        

    demo.launch(server_name="0.0.0.0", server_port=out_args.port)