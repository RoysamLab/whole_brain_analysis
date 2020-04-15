import os
import pandas as pd


def create_script(script_name, input_dir, crop_position, brightfield=None):
    """
    :param script_name: filenmae to save the script
    :param input_dir: directory of input images
    :param crop_position: list of ['xmin', 'ymin', 'xmax', 'ymax'] crop position for inter channel parameter estimation
    :param brightfield: number of the brightfile channel
    """
    # create output directory
    if not os.path.exists(os.path.dirname(script_name)):
        os.makedirs(os.path.dirname(script_name))

    # list all files in the input directory
    files = os.listdir(input_dir)
    # get only images in the directory
    # TODO: add other extensions
    files = [file for file in files if os.path.splitext(file)[1] in ['.tif', '.tiff']]

    # create DataFrame for script
    df = pd.DataFrame(index=files, columns=['biomarker', 'intra channel correction',
                                            'inter channel correction', 'channel 1', 'channel 2', 'channel 3', 'level',
                                            'xmin', 'ymin', 'xmax', 'ymax'])
    df.index.name = 'filename'

    # set all channels for correction
    df.loc[:, ['intra channel correction', 'inter channel correction']] = 'Yes'

    # remove brightfield from correction
    if brightfield:
        # TODO: add other extensions
        bf_filenames = [file for file in files if 'C' + str(brightfield) + '.tif' in file]
        df.loc[bf_filenames, ['intra channel correction', 'inter channel correction']] = 'No'
        df.loc[bf_filenames, 'biomarker'] = 'brightfield'

    # add crop position as 'xmin' 'ymin' 'xmax' 'ymax'
    df.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']] = crop_position

    # save file
    df.to_csv(script_name)


if __name__ == '__main__':
    input_dir = r'/brazos/roysam/datasets/TBI/G4_mFPI_Li+VPA/G4_BR#16_HC_10R/original'
    brightfield = 11
    crop_position = [34000, 8000, 44000, 15000]
    create_script('sample_script.csv', input_dir, crop_position, brightfield=brightfield)
