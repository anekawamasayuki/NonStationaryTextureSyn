import os.path
from data.base_dataset import BaseDataset, get_transform, get_half_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
from pdb import set_trace as st
import random
from torchvision.transforms.functional import crop


ImageFile.LOAD_TRUNCATED_IMAGES = True


class HalfDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')
                
        B_tensor = self.transform(B_img)
        # w, h = B_img.size
        # rw = random.randint(0, w - self.fineSize)
        # rh = random.randint(0, h - self.fineSize)
        # # print(rw, rh)
        # B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

        # _, w, h = B_img.size()
        # rw = random.randint(0, int(w/2))
        # rh = random.randint(0, int(h/2))

        div_by_4 = self.fineSize // 4
        div_by_2 = self.fineSize // 2

        A_tensor = crop(B_tensor, div_by_4, div_by_4, div_by_2, div_by_2)

        # A_img = self.transform(A_img)
        # B_img = self.transform(B_img)

        return {'A': A_tensor, 'B': B_tensor,
                'A_paths': path, 'B_paths': path,
                'A_start_point':[(div_by_4, div_by_4)]}

    def __len__(self):
        return self.size

    def name(self):
        return 'HalfDataset'
