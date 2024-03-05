import numpy as np
import torch
import torchvision
from torch import Tensor
import unittest as unit
import losses as l

class TestPaCMAPLoss(unit.TestCase):
    def setUp(self):
        # simulate encoder output
        return
    
    def test_l2_norm(self):
        # test PaCMAPLoss.l2_norm()
        point_1 = Tensor([7, 4, 3])
        point_2 = Tensor([17, 6, 2])
        true_answer = 10.246951
        real_answer = l.PaCMAPLoss.l2_norm(point_1, point_2)
        self.assertEqual(true_answer, real_answer)
        return
    
    #def test_pacmap_distance(self):
        # test PaCMAPLoss.pacmap_distance()
        return
    
    #def test_pacmap_near_loss(self):
        # test PaCMAPLoss.pacmap_near_loss()
        return 
    
    #def test_pacmap_midnear_loss(self):
        # test PaCMAP.pacmap_midnear_loss()
        return
    
    #def test_pacmap_far_loss(self):
        # test PaCMAPLoss.pacmap_far_loss()
        return
    
    #def test_phase_1(self):
        return
    
    #def test_phase_2(self):
        return
    
    #def test_phase_3(self):
        return
    
    #def test_forward(self):
        return

if __name__ == '__main__':
    unit.main()