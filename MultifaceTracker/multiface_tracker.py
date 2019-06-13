import os
import numpy as np
import cv2
import json
import pickle
from scipy.optimize import linear_sum_assignment
from PyPCN import *
from collections import deque
from bidict import bidict
import sys
from ipdb import set_trace as dbg

def debug_msg(string):
    print(string)


class NameGenerator():
    def __init__(self,filename):
        self.names_db = []
        self.ID=0
        with open(filename) as name_file:
            for line in name_file:
                name, _, cummulative, _ = line.split()
                self.names_db.append(name)
        pass
    def get_full_name(self):
        current_name = self.names_db[self.ID]
        self.ID = 0 if self.ID >= len(self.names_db) else self.ID + 1
        return current_name

class IDMatchingManager():
    def __init__(self,classifier_path,th_similar,th_symmetry,max_desc_len):
        self.history_ids = {}
        self.first_appearance = {}
        self.revese_matches = bidict()
        self.th_similar = th_similar
        self.th_symmetry = th_symmetry
        self.max_desc_len = max_desc_len
        self.cycle_counter = 0
        self.name_gen = NameGenerator("./dist.male.first")

        with open(classifier_path, 'rb') as fd:
            self.classifier_MLP = pickle.load(fd)
    
    def compare_descriptors(self,desc1,desc2,th_sym):
        best_match = -1.0
        for d1 in desc1:
            for d2 in desc2:
                test_vec = np.concatenate((d1,d2))
                match_prob1 = self.classifier_MLP.predict_proba([test_vec])[:,1]
                match_prob2 = self.classifier_MLP.predict_proba([np.fft.fftshift(test_vec)])[:,1]
                if np.abs(match_prob1-match_prob2) > th_sym:
                    tracked_match = -1.0 ## No decision
                tracked_match = (match_prob1+match_prob2)*0.5
                best_match = tracked_match if tracked_match>best_match else best_match
        return best_match

    def preload_ids(self,json_file):
        with open(json_file, 'r') as fp:
            self.history_ids = json.load(fp)
        
    def save_ids(self,json_file):
        temp_ids = {k:list(v) for k,v in self.history_ids.items()}
        with open(json_file, 'w') as fp:
            json.dump(temp_ids,fp,indent = 4)

    def update_matches(self,tracked_ids):

        hist_keys = list(self.history_ids.keys())
        hist_desc = list(self.history_ids.values())

        tracked_keys = list(tracked_ids.keys())
        tracked_desc = list(tracked_ids.values())

        corr_mtx = np.zeros((len(hist_keys),len(tracked_keys)))
        for idx_hist, desc_hist in enumerate(hist_desc):
            for idx_tracked,desc_tracked in enumerate(tracked_desc):
                match_prob = self.compare_descriptors(desc_hist,[desc_tracked],self.th_symmetry)
                corr_mtx[idx_hist,idx_tracked] = match_prob
        row_ind, col_ind = linear_sum_assignment(-corr_mtx)

        if self.cycle_counter == 29:
            disc1 = self.history_ids["JAMES"]
            disc2 = self.history_ids["MICHAEL"]
            test_match = self.compare_descriptors(disc1,disc2,0.02)
            dbg()
       
        # Generate tracked faces
        revese_matches = bidict()
        undecided = []
        key_pairs_for_merge = []
        for r,c in zip(row_ind,col_ind):
            if corr_mtx[r,c] > self.th_similar: 
                ## In case of duplication we want to merge history
                if tracked_keys[c] in self.revese_matches and \
                        hist_keys[r] != self.revese_matches[tracked_keys[c]]: 
                    key_pairs_for_merge.append((hist_keys[r],self.revese_matches[tracked_keys[c]]))
                revese_matches[tracked_keys[c]] = hist_keys[r]
            else:
                debug_msg("{0}: Undecided {1:.2f}".format(self.cycle_counter,corr_mtx[r,c]))
                undecided.append((r,c))

        ## Assign tracked faces since there are mote face than db
        ##TODO: Make sure unassigned are actually new numbers!!
        unassigned = np.delete(np.arange(0,len(tracked_keys),1),col_ind).tolist()
        undecided_c = [u[1] for u in undecided]

        for c in unassigned + undecided_c:
            if tracked_keys[c] in self.revese_matches: ##TODO:consider hgistory
                ##TODO: What happens if the hist is already assigned!!?
                self.history_ids[self.revese_matches[tracked_keys[c]]].append(tracked_desc[c]) #found a tracked desc of the face
                revese_matches[tracked_keys[c]] = self.revese_matches[tracked_keys[c]] #maintain reverse matching
            else: ## A totally new face
                assigned_key = self.name_gen.get_full_name()
                self.history_ids[assigned_key] = deque([tracked_desc[c]],self.max_desc_len)
                revese_matches[tracked_keys[c]] = assigned_key

        ## Perform merges 
        for a_key,b_key in key_pairs_for_merge:
            if self._get_first_appearance(a_key) < self._get_first_appearance(b_key):
               base_key,other_key = a_key, b_key
            else:
               base_key,other_key = b_key, a_key
            debug_msg("{0}: Merging {1} -> {2}".format(self.cycle_counter,other_key,base_key))

            if other_key in revese_matches.inverse:
                if base_key in revese_matches.inverse:
                    debug_msg("{0}: Bad merge:Two images matched to same merge!!!".format(self.cycle_counter))
                    continue ## Can't do merge in this situation
                else:
                    tracked_key = revese_matches.inverse[other_key]
                    revese_matches[tracked_key] = base_key

            ## Remove key from history
            other_desc = self.history_ids.pop(other_key)
            self.history_ids[base_key].extend(other_desc)

        self.revese_matches = revese_matches
        self._update_history_counter()
                   
    def _update_history_counter(self):
        for k in self.revese_matches.inverse:
            if k not in self.first_appearance:
                self.first_appearance[k] = self.cycle_counter
        self.cycle_counter += 1

    def _get_first_appearance(self,k):
            if k not in self.first_appearance:
                return sys.maxint
            return self.first_appearance[k]


    def check_match(self,id_check):
        if id_check in self.revese_matches:
            return self.revese_matches[id_check]


class MultifaceTracker():
    def __init__(self,classifier_path,th_similar,th_symmetry,max_desc_len,*args):
        self.ids_manager = IDMatchingManager(classifier_path,th_similar,th_symmetry,max_desc_len)
        self.pcn_detector = PCN(*args)


    def _normalize_desc(self,desc):
        desc = np.array(desc)
        desc /= np.linalg.norm(desc) #performane improves with normalization
        return desc.tolist() #save lists to allow json serialization

    def track_image(self,img):
        pcn_faces = self.pcn_detector.DetectAndTrack(img)

        if self.pcn_detector.CheckTrackingPeriod()==self.pcn_detector.track_period:
            tracked_ids = {pcn_det.id:self._normalize_desc(pcn_det.descriptor) for pcn_det in pcn_faces}
            self.ids_manager.update_matches(tracked_ids)

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
    #classifier_path = "./model/trained_QuadraticDiscriminantAnalysis_model.clf"

    mface = MultifaceTracker(
            classifier_path,
            0.7,0.02,50,
            detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
            tracking_model_path,tracking_proto, 
            embed_model_path, embed_proto,
            40,1.45,0.5,0.5,0.98,30,0.9,1)
    #if os.path.isfile("./tracking.json"):
    #    mface.ids_manager.preload_ids("./tracking.json")

    cap = cv2.VideoCapture("./test_tracked.mp4")
    #cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print (width,height)
    fps = cap.get(cv2.CAP_PROP_FPS) # float
    writer = cv2.VideoWriter("tracked.mp4", fourcc, fps,(width,height),True)
    while cap.isOpened():
        ret, img = cap.read()
        if img is None or img.shape[0] == 0:
            break
        #writer.write(img)

        
        try:
            faces = mface.track_image(img) 
        except:
            pass
            #writer.release()
            #exit()

        for face in faces:
            name = mface.ids_manager.check_match(face.id)
            if name is None:
                name = "Undecided"
            PCN.DrawFace(face,img,name)
            PCN.DrawPoints(face,img)

        cv2.putText(img,str(mface.ids_manager.cycle_counter),(10,80), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,255,0),3,cv2.LINE_AA)
        cv2.imshow('window', img)
        mface.pcn_detector.CheckTrackingPeriod()
        writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    mface.ids_manager.save_ids("tracking.json")
    cap.release()
    writer.release()
