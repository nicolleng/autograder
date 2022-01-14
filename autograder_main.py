import tkinter as tk
from functools import partial
from tkinter import Frame, Scrollbar, Listbox, filedialog
from math import floor, ceil
from PIL import Image, ImageTk
from numpy import argmax
from matplotlib.ticker import MaxNLocator
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras_preprocessing.image import img_to_array, load_img
from CNN_model import load_model,run_test_harness


root = tk.Tk()
root.title("Automatic Grading Application")
root.geometry("800x400")


frame = tk.Frame(root, bg="alice blue")
frame.pack(fill="both", expand=True, padx=20, pady=20)

lblstep1 = tk.Label(frame, text="Step 1", font="Arial 12", bg="alice blue")
lblstep1.place(x=60, y=50)

lblstep2 = tk.Label(frame, text="Step 2", font="Arial 12", bg="alice blue")
lblstep2.place(x=510, y=50)

lblstep3 = tk.Label(frame, text="Step 3", font="Arial 12", bg="alice blue")
lblstep3.place(x=300, y=285)



#container to hold the key image/s
KeyImageArr =[]

global KeyDigit


#containers to hold student images and names of the images
studentFileArray = []
studentFilepath = []
keyEntry=tk.Entry(frame, width=5)
keyTxt = tk.StringVar(frame)
#method to permit user to select an image which will be OCR'd and resulting digit stored as key
def chooseKeyFile():


    keyFile = filedialog.askopenfilename(initialdir='C:\\Users\\*\\Pictures',title="Select File",
                                         filetypes=(("images","*.jpg *.png"),("all files","*.*")))

    if keyFile is not None:

        keyImg =cv2.imread(keyFile)
        KeyImageArr.clear()
        KeyImageArr.append(keyImg)

        # Create a thumbnail (100x150) image of chosen key file
        imageOfKey = Image.open(keyFile)
        resized_Img = imageOfKey.resize((100,150))
        tkImageOfKey = ImageTk.PhotoImage(resized_Img)


        #place the thumbnail of the key
        labelKeyImage = tk.Label(image=tkImageOfKey)
        labelKeyImage.image = tkImageOfKey
        labelKeyImage.place(x=60, y=130)


        #grab key image and process it.
        keyImg = KeyImageArr[0]
        inputImage = remove_lines(keyImg)
        process_image(inputImage)
        bbox_implement()
        # run a prediction on the key image
        digitPredicted = run_prediction('Images\\resized.jpg')

        # store the key digit in a global variable to compare it to student answers later

        KeyDigit = digitPredicted


        keyTxt.set("AI prediction: " + str(KeyDigit))

        lblKey = tk.Label(frame, textvariable=keyTxt, font=('Arial', 12,'bold'))
        lblKey.place(x=45, y=300)


        keyEntry.place(x=65, y=270)
        keyEntry.insert(0, KeyDigit)

        btnSet = tk.Button(frame, height=1, width=8, text="Change", command=partial(clickChange,keyEntry,keyTxt))
        btnSet.place(x=85, y=270)

def clickChange(entry,keyTxt):
    newInt = int(entry.get())
    KeyDigit = newInt
    keyTxt.set("Key digit set to: " +str(KeyDigit))
    return KeyDigit

def chooseStudentAnswerFiles():


    studentFiles = filedialog.askopenfilenames(
        initialdir='C:\\Users\\*\\Pictures',title="Select Student Files",
        filetypes=[("images", "*.jpg *.png")])

    numImages = len(studentFiles)


    label = tk.Label(frame, text=str(numImages)+" student images have been selected.", bg="light green", pady = 5)
    label.place(x=40, y =335)


    #create a frame to display student images
    frame2 = tk.Frame(frame, bg="alice blue")
    frame2.place(x=450, y=80)


    a=0
    b=0
    for filepath in studentFiles:
        studentFilepath.append(filepath)
        img = cv2.imread(filepath)
        studentFileArray.append(img)
        answerImage = Image.open(filepath)
        resized_Img = answerImage.resize((25,40))

        answerImage = ImageTk.PhotoImage(resized_Img)

        labelanswerImage = tk.Label(frame2, image=answerImage)
        labelanswerImage.image = answerImage


        labelanswerImage.grid(column=b, row=a)
        a = floor(len(studentFileArray)/8)
        b = (len(studentFileArray) % 8)

    return studentFileArray



def run_prediction(outputJPG):

    img = load_image(outputJPG)
    # load model
    model = load_model()
    # predict the class
    predict_value = model(img)
    digit = argmax(predict_value)
    #print("Prediction: ", digit)
    return digit


# - Destroy buttons because the program is only capable of doing one prediction round at a time w/o breaking.
# - Predict what each image in the student image array is and add it to the student answer array.
# - Pass the answer array to the showresultswindow function and initiate it

def predict_and_evaluate():
    btnEvaluate.destroy()
    btnChooseKey.destroy()
    btnChooseStudentFile.destroy()

    # if there are files in the studentFileArray, predict them one by one and store them in an answer array
    studentAnswerArr =[]

    if len(studentFileArray) > 0:

        for img in studentFileArray:
            #process student images to match MNIST training set
            inputImage = remove_lines(img)
            process_image(inputImage)
            bbox_implement()
            #run through prediction model
            studentAnswerPrediction = run_prediction('Images\\resized.jpg')
            #record prediction
            studentAnswerArr.append(studentAnswerPrediction)
            #print("Student answer prediction: ", studentAnswerPrediction)


    show_results_window(studentFilepath,studentAnswerArr)


def show_results_window(studentFilepath,studentAnswerArr):
    KeyDigit=clickChange(keyEntry,keyTxt)
    result_window = tk.Toplevel(root)
    result_window.title("Evaluation Results")
    result_window.geometry("510x510")
    frame3=Frame(result_window)
    scrollbar = Scrollbar(result_window)
    scrollbar.pack(side='right', fill='both')

    #Create a listbox widget in the frame
    listbox = Listbox(frame3, width=500, height=350)
    listbox.config(yscrollcommand = scrollbar.set)
    scrollbar.config(command = listbox.yview)


    #compare student answer predictions to key prediction and output result
    i = 0
    numCorrect = 0
    numWrong = 0
    wrongAnswers = {}


    for file in studentFilepath:

        #if answer is correct, add to listbox with green background and increment numCorrect counter
        if studentAnswerArr[i] == KeyDigit:
            listbox.insert(i+1, file +"          AI prediction: "+ str(studentAnswerArr[i]))
            listbox.itemconfig(i,{'bg':'light green'})
            numCorrect+=1
        else:
            listbox.insert(i+1, file +"          AI prediction: "+ str(studentAnswerArr[i]))
            listbox.itemconfig(i,{'bg':'tomato'})
            numWrong+=1
            #record what the wrong answers are in the wrongAnswers dictionary
            if studentAnswerArr[i] in wrongAnswers:            #if this wrong answer already exists, add 1
                wrongAnswers[studentAnswerArr[i]] +=1
            else:
                wrongAnswers[studentAnswerArr[i]] = 1          #else add this wrong answer to the dictionary of
                #wrong answers with a count of 1
        i+=1


    str1 = "     Total: "+ str(len(studentAnswerArr))+ "      Correct: "+str(numCorrect)+"      Incorrect: "+str(numWrong)

    listbox.insert(i,str1)
    listbox.itemconfig(i,{'bg':'sky blue'})
    listbox.pack(pady=20, padx=10)


    #plot data on a pie chart
    fig = plt.figure(figsize=(6,6),dpi=100)
    fig.set_size_inches(6,4)

    labels='Correct','Incorrect'
    sizes=[numCorrect,numWrong]
    colors=['yellowgreen','tomato']
    explode=(0.2,0)

    plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', shadow=True, startangle=108)
    plt.axis('equal')
    plt.title("Percentage of Correct Responses")

    #prepare wrongAnswers to plot on bar graph
    listX = []
    for key in wrongAnswers.keys():
        listX.append(key)


    listY = []
    for value in wrongAnswers.values():
        listY.append(value)

    fig, ax = plt.subplots()

    plt.bar(listX,listY,align="center",alpha=.5)
    ax.set_xticks(listX)
    ax.set_xticklabels(listX)
    ax.set_xlabel('Student Answers')
    ax.set_ylabel('Number of Responses')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Incorrect Responses')
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('bar_plot.png')


    frame3.pack(padx=5,pady=5)
    plt.show()






#removes most colors other than black/dark gray from image
def remove_lines(image):

    #save the image so it can be changed to PIL image
    cv2.imwrite('Images\\temp_img.jpg', image)
    pil_image = Image.open('Images\\temp_img.jpg')
    image_data = pil_image.load()

    height,width = pil_image.size

    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image_data[loop1,loop2]
            # turn any pixel that's not dark grey or black into white
            if (r > 70) and (b > 45 ) and (g > 40) :
                image_data[loop1,loop2] = 255,255,255


    pil_image.save('Images\\lines_removed.jpg')
    image2 = cv2.imread('Images\\lines_removed.jpg')


    #image.show()
    return image2

#load the de-lined image and apply blurring techniques to reduce noise and smooth digit appearance
def process_image(inputImage):

    #convert BGR to grayscale
    kernel = np.ones((3,3),np.uint8)
    grayscaleImage = cv2.cvtColor(inputImage,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Images\\grayscale.jpg', grayscaleImage)
    grayscaleImage =cv2.erode(grayscaleImage,kernel, iterations=2)
    cv2.imwrite('Images\\eroded1.jpg',grayscaleImage)
    grayscaleImage = cv2.medianBlur(grayscaleImage,7)
    cv2.imwrite('Images\\medianBlur.jpg', grayscaleImage)
    otsu_threshold, image_result = cv2.threshold(grayscaleImage, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )
    cv2.imwrite('Images\\otsu.jpg', image_result)

    grayscaleImage =cv2.erode(image_result,kernel, iterations=2)
    grayscaleImage =cv2.medianBlur(grayscaleImage,7)
    grayscaleImage =cv2.dilate(grayscaleImage,kernel)
    cv2.imwrite('Images\\output.jpg',grayscaleImage)



# load processed image and prepare the image to match MNIST type/size
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

#This method grabs output.jpg from working directory, blurs, closes, invert colors, and uses contours
#to find the handwritten parts and crop them into bounding boxes. The largest one is saved as 'resized.jpg'.
def bbox_implement():

    image = cv2.imread('Images\\output.jpg')

    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


    kernel = np.ones((5,5),np.uint8)


    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('Images\\closing.jpg', closing)

    #invert the image colors
    closing=(255-closing)
    contours, hierarchy= cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)

    #iterate through the list of contours which were found
    for (i,c) in enumerate(sorted_contours):
        x,y,w,h= cv2.boundingRect(c)
        #area = cv2.contourArea(c)
        if w > 20 and h > 20:

            cropped_contour= closing[y:y+h+2, x:x+w+2]
            image_name= "Images\\croppedImage_" + str(i+1) + ".jpg"
            cv2.imwrite(image_name, cropped_contour)
            readimage= cv2.imread('Images\\croppedImage_1.jpg')

            #calculate the border size so the image is square(ish)
            if h>w:
                top, bottom, left, right = 20, 20, ceil((h+40 - w)/2), ceil((h+40 - w)/2)

            elif w>h:
                top, bottom, left, right = floor((w+40 - h)/2), floor((w+40 -h)/2), 20, 20


            resized = cv2.copyMakeBorder(
                readimage, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cv2.imwrite('Images\\resized.jpg', resized)



# entry point for training -- This might take a few hours
#run_test_harness()

#Buttons on GUI
btnChooseKey = tk.Button(root, text="Select Key Image", padx=10, pady=5, command =chooseKeyFile)
btnChooseKey.place(x=60, y=30)

btnChooseStudentFile = tk.Button(root, text="Select Student Images", padx=10, pady=5, command=chooseStudentAnswerFiles)
btnChooseStudentFile.place(x=490, y=30)

btnEvaluate = tk.Button(root, text="Evaluate", padx=10, pady=5, bg="cyan", command=predict_and_evaluate)
btnEvaluate.place(x=310, y=340)




root.mainloop()