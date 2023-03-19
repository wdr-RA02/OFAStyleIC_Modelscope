import tarfile
from .constants import *
from .train_conf import load_train_conf
from datetime import datetime as dt


def get_tar_name(dir: str):
    '''
    extracts the name of .json file for use of the tar file

    args: 
    dir: train_conf_dir

    return: name of the json file
    '''
    sep_char="\\" if "\\" in dir else "/"
    dirs=dir.strip(sep_char).split(sep_char)
    return dirs[-1].strip(".json")


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
                      generate_example: bool=False):
    '''
    Tar the model output path.

    args:
    model_path: output dir
    train_conf_path: train_conf_dir
    out_tar_name: the tar file path for output
    generate_example: whether to generate example json, with the dataset paths omitted.
    '''
    # generate example file first
    if generate_example:
        print("You choose to generate example json config rather than archiving the original file. ")
        example_path=generate_example_conf(train_conf_path)
    else:
        example_path=train_conf_path
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        with tarfile.open(out_tar_path, "w:gz") as f:
            # add config file and output folder to the archive
            f.add(model_path)
            f.add(example_path, arcname=train_conf_path)
        print("Model output dir tared to {}".format(out_tar_path))
    else:
        print("Model file does not exist, abort process")

    os.remove(example_path)

def archive_checkpoint():
    # ckpt好像也加载不进来, 就不实现了吧(bushi)
    raise NotImplementedError("checkpoint save not yet implemented")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="OFA Style model backup script")
    parser.add_argument("--conf", help="train conf file", required=True)
    parser.add_argument("-o", "--out_dir", help="dir to save the tar file", default="./work_dir/")
    parser.add_argument("-e","--example_json", help="archive example json instead of the original", action="store_true")

    args=parser.parse_args()
    # add ./ to the head of the conf file path
    train_conf_dir=args.conf
    if not args.conf.startswith("./"):
        train_conf_dir="./"+train_conf_dir
    train_conf=load_train_conf(train_conf_dir)
    current_time=dt.strftime(dt.now(), "%y%m%d-%H%M%S")
    
    model_path=os.path.join(train_conf["work_dir"], "output")
    # get the tar name and dir
    out_tar_path=os.path.join(args.out_dir, "{}_{}.tar.gz".format(get_tar_name(train_conf_dir), current_time))
    print("Out tar filename: {}".format(out_tar_path))

    archive_model_dir(model_path, 
                      train_conf_dir, 
                      out_tar_path, 
                      generate_example=args.example_json)



    
