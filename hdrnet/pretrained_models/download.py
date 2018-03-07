#!/usr/bin/env python
# encoding: utf-8

import logging
import os
import platform
import subprocess
import shutil

CHECKSUMS = {
	'pretrained_models.zip': '51636a361a6dd808a4570e89259fcb82',
}

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def main():
    files = CHECKSUMS.keys()

    url_root = "https://data.csail.mit.edu/graphics/hdrnet"

    dst = os.path.dirname(os.path.abspath(__file__))

    for f in files:
        fname = os.path.join(dst, f)
        url = os.path.join(url_root, f)
        dl_cmd = ['curl', '-o', f, url]
        if os.path.exists(fname):
            log.info('{} already downloaded. Checking md5...'.format(f))
        else:
          log.info("Running {}".format(" ".join(dl_cmd)))
          ret = subprocess.call(dl_cmd)

        if platform.system() == "Linux":
            check = subprocess.Popen(['md5sum', fname], stdout=subprocess.PIPE)
            checksum = subprocess.Popen(['awk', '{ print $1 }'], stdin=check.stdout, stdout=subprocess.PIPE)
            checksum = checksum.communicate()[0].strip()
        elif platform.system() == "Darwin":
            check = subprocess.Popen(['cat', fname], stdout=subprocess.PIPE)
            checksum = subprocess.Popen(['md5'], stdin=check.stdout, stdout=subprocess.PIPE)
            checksum = checksum.communicate()[0].strip()
        else:
            raise Exception("unknown platform %s" % platform.system())

        log.info("Checksum: {}".format(checksum, CHECKSUMS[f]))
        if checksum == CHECKSUMS[f]:
            log.info("{} MD5 correct, no need to download.".format(f))
            continue
        else:
            log.info("{} MD5 incorrect, re-downloading.".format(f))
            try:
              os.remove(fname)
            except OSError as e:
              log.info("Could not delete {}: {}".format(f, e))
              raise ValueError

            ret = subprocess.call(dl_cmd)

    log.info("Extracting files")
    zipfile = os.path.join(dst, "pretrained_models.zip")
    cmd = ["unzip", zipfile]
    subprocess.call(cmd)

    log.info("Moving directories")
    extracted_dir = os.path.join(dst, "pretrained_models")
    for d in os.listdir(extracted_dir):
      shutil.move(os.path.join(extracted_dir, d), os.path.join(dst, d))

    log.info("Cleaning up")
    os.rmdir(extracted_dir)
    os.remove(zipfile)


if __name__ == '__main__':
    main()
