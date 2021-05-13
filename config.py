INPUT_SHAPE = (352, 352, 3)
EPOCHS = 10
BATCH_SIZE = 32
VAL_RATIO = 0.15
NP_VALUE_OF_MASK = 255

OUTPUT_VIS = 'visual/'
OUTPUT_CSV = 'output.csv'


TRAIN_PATH = 'HACKATHON/images/train/'
TEST_PATH = 'HACKATHON/train.txt'
SEGMENTATION_PATH = 'HACKATHON/segmentations/'

SEMINET_PTH ='Snapshots/Semi-Inf-Net-100.pth'
SAVE_TEST = 'Results/Lung-infection-segmentation/Semi-Inf-Net/'
SAVE_TRAIN = 'Results/Lung-infection-segmentation/train/'