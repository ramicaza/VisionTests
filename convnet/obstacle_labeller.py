#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import uuid

class GetPath:
    def __init__(self,img):
        self.img = img.copy()
        self.path = []
        self.negs = []
        self.rad = 1
        self.in_pos = True
    def wait_for_pos(self):
        cv2.imshow("image",self.img)
        cv2.setMouseCallback("image", self.moused)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('f'):
                break
            elif key == ord('q'):
                exit()
            elif key == 82: #up key on my computer
                self.rad += 1
            elif key == 84: #dwn key
                self.rad -= 1
        cv2.setMouseCallback("image", lambda *args : None)
        self.in_pos = False
        return self.path, self.rad*2
    def wait_for_negs(self):
        cv2.imshow("image",self.img)
        cv2.setMouseCallback("image", self.moused)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('f'):
                break
            elif key == ord('q'):
                exit()
            elif key == 82: #up key on my computer
                self.rad += 1
            elif key == 84: #dwn key
                self.rad -= 1
        cv2.setMouseCallback("image", lambda *args : None)
        return self.negs, self.rad*2

    def moused(self,event, x, y, flags, param):
        color = (255,0,0) if self.in_pos else (0,0,255)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.rectangle(self.img,
            (x-self.rad, y-self.rad), (x+self.rad, y+self.rad),
            color, 2)
            cv2.imshow("image", self.img)
            if self.in_pos:
                self.path.append((x,y))
            else:
                self.negs.append((x,y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cpy = self.img.copy()
            cv2.rectangle(cpy,
            (x-self.rad, y-self.rad), (x+self.rad, y+self.rad),
            color, 2)
            cv2.imshow("image", cpy)

def rotate_im(image, around, angle):
  rot_mat = cv2.getRotationMatrix2D(around, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def proc_imgs(pts,outsize,img,pos):
    imh = img.shape[0]
    imw = img.shape[1]
    pad = outsize * np.sqrt(2) / 2
    filt = filter(
        lambda t : t[0] > pad and t[0] < imw-pad
        and t[1] > pad and t[1] < imh-pad
        ,pts
    )
    pts = list(filt)
    for pt in pts:
        rot = rotate_im(img,pt,np.random.randint(0,361))
        out = rot[int(pt[1]-outsize/2):int(pt[1]+outsize/2),
            int(pt[0]-outsize/2):int(pt[0]+outsize/2),:]
        out = cv2.resize(out,(64, 64), interpolation=cv2.INTER_CUBIC)
        write_path = './lanes_train' if pos else './background_train'
        write_path = "{}/{}.jpg".format(write_path,str(uuid.uuid4()))
        cv2.imwrite(write_path,out)

def main():
    in_img = sys.argv[1]
    img = cv2.imread(in_img,cv2.IMREAD_COLOR)
    gl = GetPath(img)
    print("Draw a path through the center of the lanes using your mouse...\
        Set the width of the processed images using the up and down arrows.")
    lane_pts,width = gl.wait_for_pos()
    proc_imgs(lane_pts,width,img,True)
    print("Draw a path through a bunch of places that clearly aren't lates")
    negs,width = gl.wait_for_negs()
    proc_imgs(negs,width,img,False)

if __name__ == "__main__":
    main()
