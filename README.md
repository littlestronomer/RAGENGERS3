# RAGENGERS
### Doping Hafıza - Coderspace TechAI Hackathon


```
conda create -n ragengers python=3.9
conda activate ragengers
conda install -c conda-forge boto3 ffmpeg python-dotenv botocore opencv fastapi uvicorn
pip install elevenlabs pydub python-multipart
```

`ffmpeg` is a crucial component of our project. To download it:

### Microsoft
...

### Mac
...

### Linux (Linux Mint)
```
sudo apt install ffmpeg
```

### Front End

Only `NodeJS` and `npm` (Node Package Manager) need to be installed.

## Tutorial on How to Use the Model
You need to run 
```shell
uvicorn main:app --reload
```
on your base directory

-----------------------------------------------------------------------------------------------------------------------------
The project `Doping Shorts` is designed to help students who have limited time to study or those who do not feel the pressure of deadlines but still want to practice and enhance their knowledge. It provides a useful tool for learning in flexible situations, where students can engage with the content without the stress of rigid schedules. The process begins with the user entering a query or selecting a topic of interest. Based on this input, a video is generated in the background, offering an explanation or overview of the chosen topic. After watching the video, the student can then take a quiz to test their understanding and knowledge. This quiz helps reinforce what they have learned and offers a more interactive approach to studying. Once the quiz is completed, the student can revisit the video to refresh and supplement their knowledge, creating an ongoing learning loop. This method allows for a more adaptable and self-paced study experience, making it easier for students to fit learning into their busy lives.
