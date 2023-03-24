import os
import argparse
import tarfile
import json

from datetime import datetime as dt
from utils.train_conf import load_train_conf

def get_tar_name(dir: str):
    '''
    extracts the name of .json file for use of the tar file

    args: 
    dir: train_conf_dir

    return: name of the json file
    '''
    sep_char="\\" if "\\" in dir else "/"
    dirs=dir.strip(sep_char).split(sep_char)
    return dirs[-1].replace(".json","")


def add_attr_to_conf_path(dir: str):
    '''
    add attribute '_example' to the end of conf dir

    args:
    dir: path to the config json

    return: new path with '_example' attr in it
    '''    
    sep_char="\\" if "\\" in dir else "/"
    dirs=dir.strip(sep_char).split(sep_char)
    
    dirs[-1]=dirs[-1].strip(".json")+"_example.json"

    return sep_char.join(dirs)

def generate_example_conf(train_conf_filename: str):
    '''
    replace sensitive info in conf

    args:
    train_conf: dict

    return: example_conf_file: new example json file location
    '''
    train_conf=load_train_conf(train_conf_filename)

    example_conf_dir=add_attr_to_conf_path(train_conf_filename)
    assert "img_addr" in train_conf and "dataset_path" in train_conf

    new_info={
        "img_addr": "path_to_PCap/yfcc_images/",
        "dataset_path": "path_to_PCap/personality_captions/"
    }
    train_conf.update(new_info)
    with open(example_conf_dir, "w") as f:
        json.dump(train_conf, f, indent=4)

    return example_conf_dir

def archive_model_dir(model_path: str,
                      train_conf_path: str,
                      out_tar_path: str,
                      mode: int=0):
    '''
    Tar the model output path.

    args:
    model_path: output dir
    train_conf_path: train_conf_dir
    out_tar_name: the tar file path for output
    mode: int, regarding the saving mode of json, must be one of{"original", "example", "no_save"}(0,1,2)
    '''
    # generate example file first
    if mode==0:
        example_path=train_conf_path
    elif mode==1:
        print("You choose to generate example json config rather than archiving the original file. ")
        example_path=generate_example_conf(train_conf_path)
    elif mode==2:
        print("You choose not to backup json file. ")
        example_path=None
    else:
        raise SyntaxError("mode argument is out of range, it must be one of (0,1,2), got{}".format(mode))
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        with tarfile.open(out_tar_path, "w:gz") as f:
            # add config file and output folder to the archive
            f.add(model_path)
            if example_path is not None:
                f.add(example_path, arcname=train_conf_path)
        print("Model output dir tared to {}".format(out_tar_path))
    else:
        print("Model file does not exist, abort process")
    if mode==1:
        os.remove(example_path)

def archive_checkpoint():
    # ckpt好像也加载不进来, 就不实现了吧(bushi)
    raise NotImplementedError("checkpoint save not yet implemented")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="OFA Style model backup script")
    parser.add_argument("--conf", help="train conf file", required=True)
    parser.add_argument("-o", "--out_dir", help="dir to save the tar file", default="./work_dir/")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("-e","--example_json", help="archive example json instead of the original", action="store_true")
    group.add_argument("-n","--no_backup_json", help="don't archive json file", action="store_true")
    args=parser.parse_args()
    # add ./ to the head of the conf file path
    train_conf_dir=args.conf
    if not args.conf.startswith("./"):
        train_conf_dir="./"+train_conf_dir
    train_conf=load_train_conf(train_conf_dir)
    current_time=dt.strftime(dt.now(), "%y%m%d-%H%M%S")
    model_path=os.path.join(train_conf["work_dir"], "output")

    mode=(int(args.no_backup_json)<<1)+ int(args.example_json)
    print(mode)
    # get the tar name and dir
    out_tar_path=os.path.join(args.out_dir, "{}_{}.tar.gz".format(get_tar_name(train_conf_dir), current_time))
    print("Out tar filename: {}".format(out_tar_path))

    archive_model_dir(model_path, 
                      train_conf_dir, 
                      out_tar_path, 
                      mode=mode)



    
