from utils import compute_frame_indices


def test_compute_frame_indices():
    # 1 frame per second of video
    idxs = compute_frame_indices(vid_n_frames=100, vid_fps=10, max_n_frames=10)
    assert idxs == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


    # short video, return 1 frame every second
    idxs = compute_frame_indices(vid_n_frames=100, vid_fps=30, max_n_frames=6)
    assert idxs == [0, 30, 60, 90]
    
    # long video, split into max_n_frames
    idxs = compute_frame_indices(vid_n_frames=100, vid_fps=30, max_n_frames=3)
    assert idxs == [0, 33, 66]