import re
import os
import random
import tarfile
import urllib
from itertools import islice
from torchtext import data


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle:
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


class CricketCommentaryDataset(data.Dataset):

    dirname = 'commentary_data'

    def __init__(self, text_field, label_field, label_files=('four', 'out'), max_examples=10000, path=None, examples=None, **kwargs):
        """Create an Cricket commentary dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            label_files: labels/files that will be used for this dataset. 
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            for filename in label_files:
                with open(os.path.join(path, filename), errors='ignore') as f:
                    for line in islice(f, 0, max_examples):
                        examples.append(data.Example.fromlist([line, filename], fields))
        super(CricketCommentaryDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        examples = cls(text_field, label_field, **kwargs).examples
        if shuffle:
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))
        print("dev index: %d" % dev_index)
        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))
