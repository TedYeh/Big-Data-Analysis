import cv2
import numpy as np
import random
class kmeans:
    def __init__(self, k, img_path):
        self.img_path = img_path
        self.k = k

    def calc_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def init_center(self):
        centers = []
        height, width, _ = self.img.shape
        while len(centers) < self.k:
            h, w = random.randint(0, height - 1), random.randint(0, width - 1)
            if tuple(self.img[h, w]) not in centers:centers.append(tuple(self.img[h, w]))
        return centers

    def index_to_color(self, idx):
        return self.img[idx]

    def get_new_center(self, groups):
        return [np.mean(np.array(list(map(self.index_to_color, groups[i]))), axis=0) for i in range(len(groups))]

    def cluster(self, centers):
        group = [[] for _ in range(self.k)]
        height, width, _ = self.img.shape
        
        for h in range(height):
            for w in range(width):
                mingroup = 0
                mindis = self.calc_dist(self.img[h, w], centers[0])
                for k in range(1, self.k):
                    newdis = self.calc_dist(self.img[h, w], centers[k])
                    if newdis < mindis:
                        mingroup = k
                        mindis = newdis
                group[mingroup].append((h, w))
        return group

    def run(self):
        step = 0
        self.img = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        centers = self.init_center()
        old_groups = self.cluster(centers)
        newimg = np.empty(self.img.shape, dtype=np.uint8)
        while True: 
            centers = self.get_new_center(old_groups)
            new_groups = self.cluster(centers)            
            if old_groups == new_groups:break
            old_groups = new_groups
            step += 1
            if step >= 200:break
        print(centers)
        for i in range(self.k):
            for item in new_groups[i]:
                newimg[item] = centers[i]
            
        cv2.imwrite('output_K{}.jpg'.format(self.k), newimg)
        print(step)

if __name__ == '__main__':
    img_path = input('image Path：')
    k = eval(input('clusters k：'))
    
    tmpK = kmeans(k, img_path)
    tmpK.run()
    #print(img.shape, random.choice(img[0]))