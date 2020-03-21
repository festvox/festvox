# Building a vanilla voice

### Some docker fundamentals
## One can create account on dockerhub.com. On machine, logging in is easy. Type 'docker login' on the terminal. It will prompt for username and password.
# docker pull X -> This command pulls an image. In this case, X is the name of the image
## Once we have the image locally, we can instantiate a container that runs this image
# docker run X
## To run interatively, use the flag -it
# docker run -it X
## To update the changes made in interative session, we need to commit and push. Three steps: exit the container, find container id, commit container id to image. push the image. docker ps -a lists the details
# docker commit container_name image_name
# docker push image_name
## Most scripts that are memory intense need more shared memory. We can use argument --shm-size for that
# docker run -it --shm-size 10GB X
## We can specify a name to the container for easier access later
# docker run -it --name container_name --shm-size 10GB X
## We can mount our disks onto docker
# docker run -it -v /loc_on_hdd : /loc_on_container --name container_name --shm-size 10GB X

### Docker
docker pull  srallaba/falcon:TacotronOne
nvidia-docker run -it --shm-size 10GB -v /home1:/home1  srallaba/falcon:TacotronOne
# Check that FALCONDIR variable is set
echo $FALCONDIR
# Should output $FESTVOXDIR/src/falcon where FESTVOXDIR is the location of festvox. 


### Sample build

### Building a vanilla character level model

We follow a three step procedure: Data preparation, Training and Testing

#### Data Preparation

We do the following:

1) Phones preparation
2) Acoustic Feature Extraction. We extract mel and linear spectra as acoustic features.

The idea is to use the file 'txt.phseq.data'. Then we can iterate through the files and extract linear and mel spectra. 

```text
python3.5 $FALCONDIR/utils/dataprep_addphones.py ehmm/etc/txt.phseq.data vox 
```

#### Training

```text
python3.5 local/train_phones.py --exp-dir exp/exp_tacotron_phones > log_phones 2>&1&
```

This step will create a directory called exp, a subdirectory called exp_tacotron_phones. This sub directroy will house
all the information about training. For now, it contains three folders: checkpoints, tracking and samples. Checkpoints is to store 
the checkpoints during training. The Frequency of checkpoints can be modified in the file $FALCONDIR/hyperparameters. The folder 'tracking'
will log the loss value and other information. 'samples' is intended for synthesized samples

#### Testing

```text
python3.5 local/synthesize_phones.py exp/exp_tacotron_phones/checkpoints/checkpoint_step0025000.pth ehmm/etc/txt.phseq.data.test exp/exp_tacotron_phones/samples 
```

[Samples from voices](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/tts_phseq.html)

## 

