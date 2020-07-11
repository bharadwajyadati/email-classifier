from torchtext import data
import torchtext
import torch
import os
import re
import sys

SIGNATURE_ANNOTATION = '#sig#'
REPLY_ANNOTATION = '#reply#'

SENDER_SUFFIX = '_sender'
BODY_SUFFIX = '_body'

ANNOTATIONS = [SIGNATURE_ANNOTATION, REPLY_ANNOTATION]


class ForgeProcessor(object):

    def __init__(self):
        self.input_file = "input.txt"
        self.output_file = "output.txt"

    """Checks if the file could contain message sender's name."""

    def is_sender_filename(self, filename):
        return filename.endswith(SENDER_SUFFIX)

    """By the message filename gives expected sender's filename."""

    def build_sender_filename(self, msg_filename):
        return msg_filename[:-len(BODY_SUFFIX)] + SENDER_SUFFIX

    """Builds signature detection dataset using emails from folder.

    folder should have the following structure:
    x-- folder
    |    x-- P
    |    |    | -- positive sample email 1
    |    |    | -- positive sample email 2
    |    |    | -- ...
    |    x-- N
    |    |    | -- negative sample email 1
    |    |    | -- negative sample email 2
    |    |    | -- ...

    If the dataset file already exist it is rewritten.
    """

    def build_detection_dataset(self, folder,
                                sender_known=True):

        self.build_detection_class(os.path.join(folder, u'P'),
                                   1)
        self.build_detection_class(os.path.join(folder, u'N'),
                                   -1)

    """Builds signature detection class.

    Signature detection dataset includes patterns for two classes:
    * class for positive patterns (goes with label 1)
    * class for negative patterns (goes with label -1)

    The patterns are build of emails from `folder` and appended to
    dataset file.

    >>> build_signature_detection_class('emails/P', 'train.data', 1)
    """

    def build_detection_class(self, folder,
                              label, sender_known=True):

        with open(self.input_file, 'a') as input_dataset, open(self.output_file, 'a') as output_dataset:
            input_dataset.write("input" + "~" + "output" + "\n")
            for filename in os.listdir(folder):
                filename = os.path.join(folder, filename)
                sender, msg = self.parse_msg_sender(filename, sender_known)
                if sender is None or msg is None:
                    continue
                match = re.findall(r'#sig#(.*)', msg)
                X = []
                Y = []
                if match:
                    out = match
                    msg = re.sub('|'.join(ANNOTATIONS), '', msg)
                    msg = msg.replace('\n', ' ')
                    msg = msg.replace('\t', ' ')
                    X.append(msg + "~" + "".join(out))
                    Y.append("".join(out))

                labeled_pattern = ' '.join([str(e) for e in X])
                input_dataset.write(labeled_pattern + '\n')
                labeled_pattern = ','.join([str(e) for e in Y])
                output_dataset.write(labeled_pattern + '\n')

    """Given a filename returns the sender and the message.

    Here the message is assumed to be a whole MIME message or just
    message body.

    >>> sender, msg = parse_msg_sender('msg.eml')
    >>> sender, msg = parse_msg_sender('msg_body')

    If you don't want to consider the sender's name in your classification
    algorithm:
    >>> parse_msg_sender(filename, False)
    """

    def parse_msg_sender(self, filename, sender_known=True):

        kwargs = {}
        if sys.version_info > (3, 0):
            kwargs["encoding"] = "utf8"

        sender, msg = None, None
        if os.path.isfile(filename) and not self.is_sender_filename(filename):
            with open(filename, **kwargs) as f:
                msg = f.read()
                sender = u''
                if sender_known:
                    sender_filename = self.build_sender_filename(filename)
                    if os.path.exists(sender_filename):
                        with open(sender_filename) as sender_file:
                            sender = sender_file.read().strip()
                    else:
                        # if sender isn't found then the next line fails
                        # and it is ok
                        lines = msg.splitlines()
                        for line in lines:
                            match = re.match('From:(.*)', line)
                            if match:
                                sender = match.group(1)
                                break
        return (sender, msg)


f = ForgeProcessor()
f.build_detection_dataset("forge/dataset")


data = data.TabularDataset(
    path='input.txt', format='csv', csv_reader_params={"delimiter": '~'},
    fields=[('input', data.Field()),
            ('output', data.Field())])

print(data[2].__dict__.values())
