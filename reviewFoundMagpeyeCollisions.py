import os
import cv2
import numpy as np
import shutil
from tkinter import *
from tkinter import messagebox
from globalConstants import TRAINING_EXAMPLE_DEPTH


class ImageLabeler:
    source_folder = "Pictures3D_Magpeye/"
    target_folder = "Pictures3D_Input/"


    def __init__(self, root):
        self.root = root
        self.imagesAll = os.listdir(self.source_folder) # all images, which includes the large RGB versions
        self.images = [image for image in self.imagesAll if image.startswith("Magpeye-Android_")] # only the sample images - doesn't include the large RGB versions
        if len(self.images) == 0:
            exit()
        self.current_image_index = 0
        self.image_label = None
        self.filename1_label = None
        self.filename2_label = None
        self.button_frame = Frame(root)
        self.button_frame.pack()
        self.load_image()
        self.buttons = []
        for i in range(4):
            button = Button(self.button_frame, text=f"Button{i}", command=lambda idx=i: self.on_button_click(idx))
            button.pack(side=LEFT, padx=5)
            self.buttons.append(button)
        button = Button(self.button_frame, text=f"ButtonDiscard", command=lambda idx="Discard": self.on_button_click(idx))
        button.pack(side=LEFT, padx=5)
        self.buttons.append(button)
        

    def load_image(self):
        self.imageName = self.images[self.current_image_index]
        self.imageNameParts = self.imageName.split('_')
        if len(self.imageNameParts) != 3 or self.imageNameParts[0] != "Magpeye-Android":
            print(self.imageName, "is not a valid image name.")
            exit()
        self.imageName_large = "Magpeye-Android-Large_" + self.imageNameParts[1] + "_" + self.imageNameParts[2]
        image_path = os.path.join(self.source_folder, self.imageName)
        image_path_large = os.path.join(self.source_folder, self.imageName_large)
        self.image = np.load(image_path)
        self.image_mv = np.load(image_path)
        self.image_large = np.load(image_path_large)
        newimg1 = cv2.hconcat([self.image[i,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
        newimg2 = cv2.hconcat([self.image[i+TRAINING_EXAMPLE_DEPTH//2,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
        newimg = cv2.vconcat([newimg1, newimg2])
        newimg1_large = cv2.hconcat([self.image_large[i,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
        newimg2_large = cv2.hconcat([self.image_large[i+TRAINING_EXAMPLE_DEPTH//2,:,:] for i in range(TRAINING_EXAMPLE_DEPTH//2)])
        newimg_large = cv2.vconcat([newimg1_large, newimg2_large])
        newimg_large = newimg_large[:, :, [3, 2, 1, 0]]
        #newimg_large = np.uint8(newimg_large)
        #newimg_large_rgb = cv2.cvtColor(newimg_large, cv2.COLOR_GRAY2RGBA)


        for i in range(self.image_mv.shape[0]):
            self.image_mv[i] = cv2.flip(self.image_mv[i], 1)
        photo = PhotoImage(data=cv2.imencode('.png', newimg_large)[1].tobytes())
        if self.image_label:
            self.image_label.destroy()
        if self.filename1_label:
            self.filename1_label.destroy()
        if self.filename2_label:
            self.filename2_label.destroy()
        self.image_label = Label(self.root, image=photo)
        self.image_label.image = photo
        self.image_label.pack()
        self.filename1_label = Label(self.root, text=self.imageNameParts[1])
        self.filename1_label.pack()
        self.filename2_label = Label(self.root, text=self.imageNameParts[2])
        self.filename2_label.pack()
        
    
    def on_button_click(self, button_index):
        button_index_mv = "Discard" if button_index == "Discard" else "1" if button_index == 3 else "2" if button_index == 2 else "3" if button_index == 1 else "0"

        target_subfolder = os.path.join(self.target_folder, str(button_index))
        target_subfolder_mv = os.path.join(self.target_folder, button_index_mv)

        source_image_path = os.path.join(self.source_folder, self.imageName)

        self.imageName = self.imageNameParts[0] + "_" + self.imageNameParts[1] + "_" + str(button_index) + ".npy"
        self.imageName_mv = self.imageNameParts[0] + "_" + self.imageNameParts[1] + "_mv_" + button_index_mv + ".npy"

        target_image_path = os.path.join(target_subfolder, self.imageName)
        target_image_path_mv = os.path.join(target_subfolder_mv, self.imageName_mv)

        shutil.move(source_image_path, target_image_path)

        if button_index_mv != "Discard":
            np.save(target_image_path_mv, self.image_mv)
        
        self.current_image_index += 1
        if self.current_image_index < len(self.images):
            self.load_image()
        else:
            messagebox.showinfo("Information", "Alle Bilder verarbeitet.")


def main():
    root = Tk()
    root.title("Image Labeler")
    ImageLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()