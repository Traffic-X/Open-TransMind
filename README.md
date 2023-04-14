English | [简体中文](README_ch.md)

# Introduction  

As a world-class academic summit in the field of computer vision and pattern recognition, CVPR is not only an academic conference for scholars to showcase cutting-edge scientific and technological achievements, but also a major platform for enterprises to explore cutting-edge applications. In recent years, with the explosive development of large model technology, innovative applications based on large model technology are gradually releasing huge value space in the industry. As a leader and deep cultivator in the field of artificial intelligence technology, Baidu has strong technical advantages and profound technological accumulation in the field of large model technology. As of November 2022, Baidu's independently developed industry-level knowledge enhancement large model system, the Wenxin large model, has included 36 large models, covering three levels of systems: basic large model, task large model, and industry large model, fully meeting the needs of industrial applications, The largest industrial model system in the industry has been constructed. As one of the cores of the Wenxin foundation Model, the Wenxin · CV foundation Model VIMER has been widely used in core businesses such as autonomous driving, cloud intelligence integration, and mobile ecology.


In order to further promote the development of visual large model technology, Baidu will hold the first large model workshop on CVPR 2023 this year, inviting top scholars and elites in the field of large models to jointly discuss the current situation and future of large model technology. At the same time, it will hold the first international competition for multitasking large models in the field of intelligent transportation, providing a platform for exchange and exchange of large model application technology. On March 28, 2023, we will officially launch the first International Competition for Large Model Technology, opening the registration channel to developers worldwide. (See the end of the article for the address of the competition)


In this  model technology competition, we aim at the direction of intelligent transportation and open source Open-TransMind v1.0 as a baseline for competitors, providing a great opportunity for global challengers to exchange cutting-edge foundation model technology.


About Open TransMind v1.0

In 2022, Baidu proposed the Unified Feature Optimization (UFO) technology, and released the world's largest visual model VIMER-UFO 2.0 (Wenxin · CV foundation Model), covering 20+CV basic tasks, achieving 28 public data sets SOTA. Subsequently, Baidu Apollo combined UFO technology and intelligent transportation AI capabilities into one of the multimodal, multi-scene, and multitasking Wenxin Transportation foundation Models ERNIE-Traffic-TransMind, It can simultaneously support three modes: point cloud, vision, and text, including over a hundred traffic features in multiple scenarios such as autonomous driving, vehicle road collaboration, intelligent traffic management, intelligent Internet connection, intelligent parking, and intelligent highway. It has pioneered the introduction of the open world understanding ability of text image dialogue and text image mode conversion ability, and has been gradually applied to various solutions and product lines of Baidu Intelligent Transportation.


1、 Competition background

Double Track Challenge Upgrade Explore the Way to Innovate Large Model Technology


In recent years, the development of industries such as smart cars and artificial intelligence has created good development opportunities for the development of intelligent transportation. Intelligent transportation related technologies have penetrated into our daily lives, but the multitasking processing mode of existing large models and traditional perceptual methods (such as classification, detection, segmentation, etc.) cannot meet our pursuit of wider traffic scenarios and higher levels of autonomous driving. Starting from the key issues in current practical technical research, we have set up two major tracks:


Track 1: Solving conflicts between multiple tasks and data

Previously, the mainstream visual model production process typically used a single task "train from scratch" scheme. Each task is trained from scratch, and there is no way to learn from each other. Due to the bias problem caused by insufficient single task data, the actual effect relies excessively on the distribution of task data, and the scene generalization effect is often poor. The booming big data pre training technology in the past two years has learned more general knowledge through using a large amount of data, and then migrated to downstream tasks. Essentially, different tasks have borrowed from each other's knowledge. The pre training model based on massive data acquisition has good knowledge completeness, and fine tuning based on small amounts of data can still achieve good results in downstream tasks. However, the model production process based on pre training+downstream task fine tuning requires training models for each task, resulting in significant R&D resource consumption.

The VIMER-UFO All in One multitask training scheme proposed by Baidu can be directly applied to processing multiple tasks by training a powerful general model using data from multiple tasks. Not only improves the effectiveness of a single task through cross task information, but also eliminates the downstream task fine-tuning process. The VIMER-UFO All in One R&D model can be widely used in various multitasking AI systems. Taking a smart city scene as an example, VIMER-UFO can use a single model to achieve SOTA effects for multiple tasks such as face recognition, human body, and vehicle ReID. At the same time, the multitask model can achieve significantly better effects than the single task model, demonstrating the effectiveness of the information reference mechanism between multiple tasks.


Track 2: Understanding and Perception of Scene Text Images

High performance image retrieval capabilities in traffic scenes are very important for traffic law enforcement and public security governance. Traditional image retrieval methods typically use attribute recognition of images before achieving retrieval capabilities by comparing them with desired attributes. With the development of multimodal large model technology, the representation unification and modal transformation of text and images have been widely used. Using this ability can further improve the accuracy and flexibility of image retrieval.

**The winner teams are invited to present their solutions on [CVPR 23 foundation model workshop](https://foundation-model.com/)) in person.   Besides, Top 3 teams of each track are invited to submit papers  without using the cmt system（only for extended abstract paper）. Please refer to the [paper submisison page](https://foundation-model.com/Paper_Submission) for more details .**

# Schedule
The competition is divided into two independent tracks, and contestants can choose at will or participate in both tracks at the same time.  Click [ here](https://aistudio.baidu.com/aistudio/competition/detail/848/0/introduction) to join Track 1.
| Timeline | Schedule | 
| -------- | -------- |
| 2023/3/28 00:00:00 | Sign up and Release dataset  |
| 2023/4/1 00:00:00| Release leaderboard A |
| 2023/4/8 00:00:00| Release baseline (PaddlePaddle Version)|
| 2023/5/17 EST 23:59:59（GMT+8 5/18 14:59）| Close sign-up and Close  leaderboard A |
| 2023/5/19 23:59:59| Release leaderboard B |
| 2023/5/20 00:00:00 -2022/5/22 23:59:59 | Submmit code (Only for top 10 teams of Leaderboard B)  |
| 2023/5/23 00:00:00-2022/6/6 23:59:59 | Review submmit code and reproduce the results|   
| 2023/6/7 12:00:00 |  Release final leaderboard|   
| 2023/6/12 23:59:59 | Submmit vedios (Only for top 3 teams)   |   
| 2023/6/19 | CVPR 2023 foundation model workshop |   

# Competition Bonus
The total prize pool of this competition is 10000 US dollars, and each track is 5000 US dollars. Only PaddlePaddle solutions can get  bonus. 
      
|  Award | Quantity | Bonus |
| -------- | -------- | -------- |
| First prize  | 1   | 2500 US dollars  |
| Second prize  | 1   | 1500 US dollars |
| Third prize  | 1   | 100 US dollars  |
      
**Note:**  The top 10 teams in the final ranking list can only get the corresponding bonus if they use PaddlePaddle framework and agree to open source, and the award ranking will not be postponed (if they do not use PaddlePaddle framework, the ranking will not be cancelled, but they will not get the bonus).
   
 # Participants and requirements
**Participants:**  
The competition is open to the whole society, regardless of age, identity and nationality. Individuals in related fields, universities, scientific research institutions, enterprises and start-up teams can sign up for the competition.  Baidu employees can sign up to participate, but cannot win the prize.  
  
**Entry requirements:**  
Individual or team participation is supported. The maximum number of participants in each team is no more than 15. Cross units are allowed to form teams freely, but each team can only participate in one team.  

 # Competition  rules
(1) All contestants must register in **AI studio platform** ; 
(2) Contestants should ensure that the information submitted during registration is accurate and valid, and all qualifications and bonus payments are subject to the information submitted;  
(3) Contestants can form teams on the "my team" page after they sign up. Each team needs to appoint a team leader, and the total number of team members **shall not exceed 10**. Each contestant can only participate in one team. Once a contestant is found to participate in multiple teams by registering multiple accounts, the qualification of relevant teams will be cancelled;  
(4) The name of the team shall not be set in violation of Chinese laws and regulations or public order and good customs, and the words "Baidu official", "feioar official", "paddle official", "official baseline" and so on shall not appear in the name of participating teams. If the name of the team is not changed after receiving the warning from the organizer, the organizer has the right to dissolve the team;  
(5) Except for the data set provided by the organizer, contestants are not allowed to use the marked data from any other channels;  
(6) The teams can upload the forecast results of the test set at any time during the competition. In the stage a, the team can be evaluated 5 times a day at most. The competition management system will update the current highest score and the latest ranking of each team in real time;  
 
# Anti cheating instructions

(1) Participants are not allowed to register for multiple accounts. Once found, their scores will be cancelled and dealt with seriously.  
(2) Participants are not allowed to use rules loopholes or technical loopholes and other bad ways to improve their performance ranking outside the scope of technical ability assessment. Once found, their performance will be cancelled and dealt with seriously.  
(3) AI Studio will collect player information, code, model and system report for performance evaluation, competition notice and other related competition matters.
