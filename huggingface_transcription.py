from transformers import pipeline
from datetime import datetime
import argparse
import ffmpeg
import os
import subprocess
import shlex
import psycopg2


def transcribe_audio(audio):
    # create the pipeline - unused
    pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-large-960h-lv60-self")
    # transcribe the audio
    result = pipe(audio)
    # return the result
    return result


# this script will transcribe audio files based on a directory of mp3 files and insert them into a postgres database
def main(source, output, converted, db, table, topic, host, dirpattern):
    netshare = source
    podfolders = os.listdir(netshare)
    mupods = [item for item in podfolders if item.startswith(dirpattern)]

    for pod in mupods:
        netfolder = pod
        netpath = os.path.join(netshare, netfolder)
        netfiles = os.listdir(netpath)
        netfilepaths = [f"{netshare}\{netfolder}\{item}" for item in netfiles]
        print(netfilepaths)
        indir = output
        outdir = converted
        # use huggingface's automatic speech recognition pipeline to transcribe the audio
        # https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline
        conn = psycopg2.connect(dbname=db, user=os.environ.get("PGUSER"), password=os.environ.get("PGPASS"), host=host,
                                port=5432)
        cur = conn.cursor()
        for nfile in netfilepaths:
            print(f"working on {nfile}")
            if nfile.endswith(".mp3"):
                # split the file into 60 second segments
                os.chdir(indir)
                # subprocess popen to split mp3 file
                command = f'ffmpeg -i {nfile} -f segment -segment_time 60 -c copy output%03d.mp3'
                print("command: ", command)
                # args = shlex.split(command)
                subprocessres = subprocess.run(command, shell=True)
                print("subprocessres: ", subprocessres)
                files = os.listdir(indir)
                transcription = list()
                for file in files:
                    fpath = os.path.join(indir, file)
                    f = ffmpeg.input(fpath)
                    outfile = os.path.join(outdir, file).replace(".mp3", ".flac")
                    f = ffmpeg.output(f, outfile, acodec='flac')
                    ffmpeg.run(f)
                    # outfile is the flac to be worked on
                    print(f"working on file: {file}")
                    # transcribe
                    pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-large-960h-lv60-self")
                    res = pipe(outfile)
                    print(res)
                    transcription.append(res['text'])
                ttext = "|".join(transcription)
                # insert into postgres transcriptions table
                sql = f"insert into {table} (topic, file, transcription, tdate) values (%s, %s, %s, %s)"
                cur.execute(sql, (topic, nfile, ttext, datetime.utcnow()))
                print(f"inserted {nfile} into transcriptions table")
                conn.commit()

                # delete the processing files
                for dir in [indir, outdir]:
                    for file in os.listdir(dir):
                        rempath = os.path.join(dir, file)
                        os.remove(rempath)
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("-s", "--source", help="source directory of mp3 files", required=True)
    parser.add_argument("-o", "--output", help="output directory of flac files", required=True)
    parser.add_argument("-c", "--converted", help="converted flac files", required=True)
    parser.add_argument("-d", "--db", help="database name", required=True)
    parser.add_argument("-t", "--table", help="table name", required=True)
    parser.add_argument("-p", "--topic", help="topic", required=True)
    parser.add_argument("-h", "--host", help="host", required=True)
    parser.add_argument("-dp", "--dirpattern", help="directory pattern", required=True)
    args = parser.parse_args()
    main(args.source, args.output, args.converted, args.db, args.table, args.topic, args.host, args.dirpattern)
