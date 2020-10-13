from __future__ import print_function, absolute_import

import os.path as osp
import glob
import re


# embed()

class Market1501(object):
    dataset_dir = 'market1501'

    def __init__(self, root = 'data', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir,'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir,'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("Path {} is not exist !".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("Path {} is not exist !".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("Path {} is not exist !".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("Path {} is not exist !".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel = False):

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            # pid == picture index
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            assert pid >= 0 and pid <= 1501
            assert camid >= 0 and camid <=6

            camid -= 1

            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        num_imgs = len(img_paths)

        return dataset, num_pids, num_imgs




# if __name__ == '__main__':
#     data = Market1501(root = '/home/qianchen/reid/data')
