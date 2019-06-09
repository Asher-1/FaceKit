from multiprocessing import Manager, Pool
import pickle
from PyPCN import *
import os

def normalize_desc(desc):
    desc = np.array(desc)
    desc /= np.linalg.norm(desc) #performane improves with normalization
    return desc.tolist() #save lists to allow json serialization

if __name__=="__main__":
    train_path = "./lfw/"
    train_dir = os.listdir(train_path)
    detection_model_path = "./model/PCN.caffemodel"
    pcn1_proto = "./model/PCN-1.prototxt"
    pcn2_proto = "./model/PCN-2.prototxt"
    pcn3_proto = "./model/PCN-3.prototxt"
    tracking_model_path = "./model/PCN-Tracking.caffemodel"
    tracking_proto = "./model/PCN-Tracking.prototxt"
    embed_model_path = "./model/resnetInception-128.caffemodel"
    embed_proto = "./model/resnetInception-128.prototxt"

    detector = PCN(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
			tracking_model_path,tracking_proto, 
                        embed_model_path, embed_proto,
			15,1.45,0.5,0.5,0.98,30,0.9,1)

    def embed(elem):
        idp,person = elem
        pix = os.listdir(train_path + person)
        print(idp,person)
        shared_dict[person] = manager.list() #must be shared list
        for person_img in pix:
            img = cv2.imread(train_path + person + "/" + person_img)
            faces = detector.Detect(img)
            if len(faces) != 1:
                continue
            shared_dict[person].append(normalize_desc(faces[0].descriptor))

    manager = Manager()
    shared_dict = manager.dict()

    ## serial version
    #for elem in enumerate(train_dir):
    #    embed(elem)
    
    #parallel version
    pool = Pool (processes=7)
    pool.map(embed, enumerate(train_dir))
    pool.close()

    persons_dict = {k:list(v) for k,v in shared_dict.items()}

    with open('persons_dict.pb', 'wb') as fd:
        pickle.dump(dict(persons_dict),fd)
