from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
from datasets import PascalVOCDataset
from collections import Counter
import threading
n_classes = len(label_map)
from model import SSD300, MultiBoxLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSD300(n_classes=n_classes)
# Load model checkpoint
model = model.to(device)
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    matrix=[]
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    #det_scores=det_scores[0].to('cpu')
    # print(det_scores)
    # print(det_boxes)
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    #matrix.extend("1")
    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    matrix.extend(det_labels)


    #matrix.extend("None")
    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image,matrix

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 100)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()

        #print(det_scores[i])
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]],width=20)
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        #text=[det_labels[i].upper(),det_scores[i]]
        #cccprint(text_size)
        #font = ImageFont.truetype("arial.ttf", fontsize)
        outline_width=1000
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]],width=500)
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
        #print(det_labels[i].upper())
    del draw
    #print("matrix: "+ str(matrix))
    return annotated_image,matrix

#'C:\Users\Iconsense\Documents\virat\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages'
import cv2
import numpy as np
from PIL import Image
import os
import sys

if __name__ == '__main__':
    file = open("report.txt", "r+")
    file.truncate(0)
    file.close()
    f = Counter()
    cap = cv2.VideoCapture('v9.mp4')
    sec=0
    q=0
    q=str(q)
    lab = []
    count = 0
    #img_arr = []
    while (True):
        #my_timer = threading.Timer(5.0, mytimer)
        #my_timer.start()
        # Capture frame-by-frame
        mat = []
        count += 1
        _, frame = cap.read()

        if frame is None:
            break
        #print(mat)
        pil_image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        img, mat = detect(pil_image, min_score=0.5, max_overlap=0.5, top_k=200)
        #pil_image = Image.open(img).convert('RGB')
        lab.extend(mat)
        open_cv_image = np.array(img)

        open_cv_image=cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        # Convert RGB to BGR
        #open_cv_image = open_cv_image[:, :, ::-1].copy()
        # scale_percent = 20 # percent of original size
        # width = int(open_cv_image.shape[1] * scale_percent / 100)
        # height = int(open_cv_image.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # resize image
        scale_percent = 40  # percent of original size
        width = int(open_cv_image.shape[1] * scale_percent / 100)
        height = int(open_cv_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image

        vid = cv2.resize(open_cv_image, dim, interpolation=cv2.INTER_AREA)
        s=Counter(lab)
        #print(s)
        window_name = 'video'

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (50, 50)

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2
        putt=q +"s"
        # Using cv2.putText() method
        vid = cv2.putText(vid, putt, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow(window_name, vid)
        os.chdir("C:/Users/Iconsense/Desktop/res_new/res_new/v9/")
        cv2.imwrite("frame-" + str(count) + ".jpg", vid)
        #open_cv_image = cv2.resize(open_cv_image, dim, interpolation = cv2.INTER_AREA)
        #cv2.imshow('frame',open_cv_image)
        # if count%50==0:

        #     os.chdir("C:/Users/Iconsense/abhishek/final/a-PyTorch-Tutorial-to-Object-Detection-master/img0")
        #     cv2.imwrite("frame-" + str(count) + ".jpg", open_cv_image)
        # #img_arr.append(frame)
        # count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count%60==0:
            fo = open("report.txt", "a",encoding='utf-8')
            #sec=int(sec)
            sec+=1

            q=str(sec)
            #print(sec)
            stat=("Time elasped:- "+ q +"\n")
            fo.write(stat)

            if "bulldozer" in s:
                if s["bulldozer"] > 1:
                    fo.write("Activity : Spread filling\n\n")
                    #fo.write("\nBulldozer is present at the construction site\n")
                    fo.write("01 50 00 Temporary Facilities and Controls\n")
                    fo.write( "‭31 23 16.17 0190 Fill, from stockpile, 300H.P. dozer, 2-1/2 C.Y., 300' haul, spread fill, sand and gravel,  1  C.Y. excavator, 14' to 20' deep,  with front end loader, excludes compaction\n\n")


            if "crane" in s:
                if s["crane"] > 1:
                    fo.write("\nCrane is present at the construction site\n")
                    fo.write("01 50 00 Temporary Facilities and Controls\n")
                    fo.write("‭01 54 19600.010 Crane crew, tower crane, static, 130' high, 106' jib, 6200 lb. capacity, monthly use, excludes concrete footing\n")
                    fo.write(" Moving materials for crews: crane\n\n")
            if "ladder" in s:
                if s["ladder"] > 1:
                    fo.write("\nActivity: Scaffolding Specialties & Staging Aid\n 01 50 00 Temporary Facilities and Controls\n")
                    fo.write("01 50 00 Temporary Facilities and Controls\n")
                    fo.write("‭01 54 23.80.2100‬ Staging Aid/Fall Protection Equip., steel side-rail ladder jack, per pair,buy\n")
                    fo.write("‭‭01 54 2375.3800‬ Scaffolding Specialties, rolling ladder with handrails, buy, 30 W x 2 step\n\n")

            if "formwork" in s:
                if s["formwork"] > 1:
                    fo.write("\nActivity: Concrete Forming \n")
                    fo.write("03 10 00 Concrete Forming and Accessories\n")
                    fo.write("‭03 11  1345.1500 Casting In Place concrete forms, footing, keyway, tapered wood, 2x4, 4 use, includes erecting, bracing, stripping, and cleaning\n")
                    fo.write("‭03 11  1345.5300 Casting in Place concrete forms, pile cap, square or rectangular, plywood,  1 use, includes erecting, bracing, stripping, and cleaning\n\n")

            if "rebars" in s:
                if s["rebars"] > 1:
                    fo.write("\nActivity : Casting in place\n")
                    fo.write("03 30 00 Cast-in-Place Concrete\n")
                    fo.write(
                        "‭03 30 5340.3850 Structural Concrete, in place, spread footing (3000 psi), over 5. C.Y., includes forms, reinforcing steel, concrete, placing and finishing\n")
                    # print("‭\t03 11  1345.5300 Casting in Place concrete forms, pile cap, square or rectangular, plywood,  1 use, includes erecting, bracing, stripping, and cleaning\n")
                    fo.write("03 31 00 Structural Concrete\n")
                    fo.write(
                        "03 31 0570.2100  Structural Concrete, placing, continuous footing, direct chute, excludes material.\n\n")


            if "excavator" in s:
                if s["excavator"] > 1:
                 #   print("sdvksdhvdksjvbsjvbsdkvbs kvsdb khbsdcjsdbckjsd sdkjvbsdkjvbsdkbdvhjjyhcfgv gfgv fhvc jyhgcgcjgdgsesfdghjhggfdzxfcgvhbjvcbvbjhutydfxcjhtdrxcvbnbjhdfxcbnhgfdx")
                    fo.write("\n Activity : Excavating and Filling \n")
                    fo.write("31 23 00 Excavation and Fill\n")
                    fo.write(
                        "31 23 16.13 3050 Excavating, trench or continues footing, common earth,  3/8  C.Y. excavator, 1' to 4' deep,  excludes sheeting dewatering\n")
                    fo.write(
                        "31 23 16.13 6250 Excavating, trench or continues footing, sand and gravel,  1  C.Y. excavator, 14' to 20' deep,  excludes sheeting dewatering\n\n")
                    #fo.write("Here the activity would be: Excavating\n\n")

            # fo.write("Python is a great language.\nYeah its great!!\n")
            s.clear()
            #print(s)
            lab.clear()
            fo.close()


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




