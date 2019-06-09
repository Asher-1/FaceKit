import os
import numpy as np
import cv2
import names
import json
import pickle
from scipy.optimize import linear_sum_assignment
from PyPCN import *

class IDMatchingManager():
    def __init__(self,classifier_path,threshold=0.5):
        self.prev_ids = {} #{"default":[0.0]*DESCRIPTORS}
        self.reverse_matched_id = {}
        self.threshold = threshold

        with open(classifier_path, 'rb') as fd:
            self.classifier_MLP = pickle.load(fd)
    
    def preload_ids(self,json_file):
        with open(json_file, 'r') as fp:
            self.prev_ids = json.load(fp)
        
    def save_ids(self,json_file):
        with open(json_file, 'w') as fp:
            json.dump(self.prev_ids,fp,indent = 4)

    def update_matches(self,new_ids):
        prev_keys = list(self.prev_ids.keys())
        prev_desc = list(self.prev_ids.values())

        new_keys = list(new_ids.keys())
        new_desc = list(new_ids.values())

        corr_mtx = np.zeros((len(prev_keys),len(new_keys)))
        for idx_prev, desc_prev in enumerate(prev_desc):
            for idx_new,desc_new in enumerate(new_desc):
                test_vec = np.concatenate((desc_prev,desc_new))
                match_prob = self.classifier_MLP.predict_proba([test_vec])[:,1]
                corr_mtx[idx_prev,idx_new] = match_prob
        row_ind, col_ind = linear_sum_assignment(-corr_mtx)
       
        # Generate new faces
        self.reverse_matched_id.clear()
        for r,c in zip(row_ind,col_ind):
            if corr_mtx[r,c] < self.threshold:
                assigned_key = names.get_full_name()
                self.prev_ids[assigned_key] = new_desc[c]
                self.reverse_matched_id[new_keys[c]] = assigned_key
            else:
                self.reverse_matched_id[new_keys[c]] = prev_keys[r]
   
        ## Assign in case more face than in db
        new_idx = np.delete(np.arange(0,len(new_keys),1),col_ind)
        for c in new_idx:
            assigned_key = names.get_full_name()
            self.prev_ids[assigned_key] = new_desc[c]
            self.reverse_matched_id[new_keys[c]] = assigned_key

    def check_match(self,id_check):
        if id_check in self.reverse_matched_id:
            return self.reverse_matched_id[id_check]

class MultifaceTracker():
    def __init__(self,classifier_path,classifier_th,*args):
        self.ids_manager = IDMatchingManager(classifier_path,threshold=classifier_th)
        self.pcn_detector = PCN(*args)


    def _normalize_desc(self,desc):
        desc = np.array(desc)
        desc /= np.linalg.norm(desc) #performane improves with normalization
        return desc.tolist() #save lists to allow json serialization

    def track_image(self,img):
        pcn_faces = self.pcn_detector.DetectAndTrack(img)

        if self.pcn_detector.CheckTrackingPeriod()==self.pcn_detector.track_period:
            new_ids = {pcn_det.id:self._normalize_desc(pcn_det.descriptor) for pcn_det in pcn_faces}
            self.ids_manager.update_matches(new_ids)

        return pcn_faces

    def detect_image(self,img):
        pcn_faces = self.pcn_detector.Detect(img)
        return pcn_faces


if __name__=="__main__":
    detection_model_path = "./model/PCN.caffemodel"
    pcn1_proto = "./model/PCN-1.prototxt"
    pcn2_proto = "./model/PCN-2.prototxt"
    pcn3_proto = "./model/PCN-3.prototxt"
    tracking_model_path = "./model/PCN-Tracking.caffemodel"
    tracking_proto = "./model/PCN-Tracking.prototxt"
    embed_model_path = "./model/resnetInception-128.caffemodel"
    embed_proto = "./model/resnetInception-128.prototxt"
    classifier_path = "./model/trained_MLPClassifier_model.clf"

    mface = MultifaceTracker(
            classifier_path,0.5,
            detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
            tracking_model_path,tracking_proto, 
            embed_model_path, embed_proto,
            40,1.45,0.5,0.5,0.98,30,0.9,1)

    if os.path.isfile("./tracking.json"):
        mface.ids_manager.preload_ids("./tracking.json")

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fps = cap.get(cv2.CAP_PROP_FPS) # float

    while cap.isOpened():
        ret, img = cap.read()
        if img.shape[0] == 0:
            break
        for face in mface.track_image(img):
            name = mface.ids_manager.check_match(face.id)
            if name is None:
                name = "Error"
            PCN.DrawFace(face,img,name)

        cv2.imshow('window', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    mface.ids_manager.save_ids("tracking.json")
