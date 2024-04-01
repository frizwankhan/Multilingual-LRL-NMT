import os
import subprocess
import shutil

languages = ["bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

# Get the absolute path of the current file
absolute_path = os.path.abspath(__file__)
directory_path = os.path.dirname(absolute_path) 

train_dataset_path = os.path.join(directory_path, "finalrepo", "train", "pmi")
dev_dataset_path = os.path.join(directory_path, "finalrepo", "dev")

final_train_directory = os.path.join(directory_path, "dataset/train_data")
os.makedirs(final_train_directory, exist_ok=True)  
final_dev_directory = os.path.join(directory_path, "dataset/dev_data")
os.makedirs(final_dev_directory, exist_ok=True)  

# Converting Training Scripts
def convert_training_data():
    for lang in languages:
        new_lang_path = os.path.join(final_train_directory, "en-"+lang)
        os.makedirs(new_lang_path, exist_ok=True)
        
        file1 = os.path.join(train_dataset_path, "en-"+lang, "train."+lang)
        file2 = os.path.join(train_dataset_path, "en-"+lang, "train.en")
        
        if lang!="hi":
            command = [
                "python", 
                "indic_scriptmap.py",
                file1,
                os.path.join(new_lang_path, "train."+lang),
                lang,
                "hi"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
        else:
            shutil.copyfile(file1, os.path.join(new_lang_path, "train.hi"))
            
        shutil.copyfile(file2, os.path.join(new_lang_path, "train.en"))

# converting dev scripts
def convert_dev_data():
    shutil.copyfile(os.path.join(dev_dataset_path, "dev.en"), os.path.join(final_dev_directory, f"dev.en"))
    for lang in languages:
        if lang=="hi":
            shutil.copyfile(os.path.join(dev_dataset_path, "dev.hi"), os.path.join(final_dev_directory, f"dev.hi"))
            continue
        
        langugage_path = os.path.join(dev_dataset_path, "dev."+lang)
        new_lang_path = os.path.join(final_dev_directory, "dev."+lang)
        command = [
                "python", 
                "indic_scriptmap.py",
                langugage_path,
                new_lang_path,
                lang,
                "hi"
            ]
        result = subprocess.run(command, capture_output=True, text=True)
        
convert_training_data()
convert_dev_data()