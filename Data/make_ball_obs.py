import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import hickle as hkl

def make_square_ball(x, y, im_rows=20, im_cols=20, width=3, height=3, noise_sig=0.1):

    T = x.shape[0]
    vid = np.zeros((T, im_rows, im_cols, 1), dtype=np.dtype(float))
    for t in range(T):
        vid[t, y[t]:y[t]+height, x[t]:x[t]+width] = 1.
        vid[t, :, :, 0] += npr.normal(0, noise_sig, (im_rows, im_cols))
    return vid

def generate_training_videos(no_videos, no_frames=10, storage_path=os.getcwd(), im_rows=20, im_cols=20):
    '''
    Generates videos of a square moving horizontally to the center and then randomly goes up or down
    '''
    gen_videos = np.zeros([no_videos, no_frames, im_rows, im_cols, 1])
    traj = np.zeros([no_videos, no_frames, 2])
    directions = np.zeros([no_videos])
    storage = {}

    for i in range(no_videos):
        x_init = npr.randint(0, im_rows/4)
        x = np.concatenate((np.arange(x_init, im_rows/2, 1), (im_rows/2-1)*np.ones(x_init)))
        top = npr.binomial(1, 0.5)
        if top:
            directions[i] = 0
            y = np.concatenate((im_cols/2*np.ones(im_rows/2-x_init), np.arange(im_cols/2-1, im_cols/2-1-x_init, -1)))
        else:
            directions[i] = 1
            y = np.concatenate((im_cols/2*np.ones(im_rows/2-x_init), np.arange(im_cols/2+1, im_cols/2+1+x_init, 1)))

        traj[i] = np.array(zip(x, y))
        storage['description']= '1000 training videos of a square 3x3 ball on 20x20 frame starting from the middle row and a random column and moving towards the center with 1 pixel/frame. When it reaches (9,9) the ball randomly moves up or down. Gaussian noise with variance 0.1 is added.'
        storage['trajectories']=traj
        gen_videos[i] = make_square_ball(x, y)
        storage['videos'] = gen_videos
        storage['directions']=directions

    f = open(storage_path, 'w')
    hkl.dump(storage, f)
    f.close()

if __name__=="__main__":

    no_videos = 200
    generate_training_videos(no_videos, storage_path='/Users/ybansal/Documents/PhD/Projects/stochastic-prednet/Data/confused_ball/test.hkl')
    
    
    
