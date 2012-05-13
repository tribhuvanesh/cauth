## Introduction

Victimized by public revelations of concocted closet secrets on social networking sites, or Machiavellian mails  to your subordinates and co-workers, after that dose of updates you crave for at moments by logging on to a publicly available system? Maybe personal system too, if your cat tends to use your authorized account to make purchases online from the Play store or itunes. Schizophrenics can rest assured knowing that they alone can be held accountable for their actions, eliminating that blame-game they love to play. 

Continuous Authentication aims to solve these mishaps, by providing an added level of security. The name, kind of self-explanatory at this point, achieves this by computing the probability that the user who is logged-in using password-based conventional techniques, is indeed the user in front of the system.

Unlike other Continuous Authentication systems, which propose to keep the user in 3 states - password login, hard biometrics and soft biometrics, we plan to eliminate the need to ask for password since this deviates the user from his work-flow. To compensate for the noise inherent in the process of face recognition, we integrated a Support Vector Machine to predict the state(Authorized or Unauthorized) given the features extracted from eigenface.

## Requirements

* g++ >= 4.5
* Python 2.7
* OpenCV >= 2.1
* Crypto++
* Linux
* libsvm 3.12 (Included)

## Usage

Compiling:

	$ bash make.sh
(Blasphemous, I know!)

Collection:

`$ ./main --collect`

Learn from data:

`$ ./main --learn`

Continuous authenticate mode:

`$ ./main`

## About

The project is being developed by Tribhuvanesh O and Soumya G, as a part our Undergraduate thesis under the guidance of Dr. K.G.Srinivasa, at M.S.Ramaiah Institute of Technology, Bangalore

## Thanks to

[Shervin Emami](http://www.shervinemami.co.cc/) for thoroughly documenting face recognition

Servo magazine for their articles on eigenfaces

[libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm/), the library for Support Vector Machines
