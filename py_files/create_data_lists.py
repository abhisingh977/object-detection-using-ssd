from utils import create_data_lists

if __name__ == '__main__':
    root1='C:/Users/Iconsense/abhishek/final/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
#   root2='C:/Users/Iconsense/abhishek/final/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    create_data_lists(voc07_path=root1,
#                      voc12_path=root2,
                      output_folder='./')
#C:\Users\Iconsense\abhishek\final\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007