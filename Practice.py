from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams

count = 0
stemmer = PorterStemmer()
englishStopwords = stopwords.words('english')

myText = "I am Surya Murali. John Cena is the best."
myText1 = "Very thorough and took time to talk to me about my problem. I would highly recommend this doctor"
myText2 = "Waiting time was the worse I have ever experience not just once every appointment at least a 3 to 4 hour wait! I would like to see things a little  more perfected in the pre-op part. I seen her once before deciding to switch to another OB/GYN. I was young and preganant at 21 years old. I felt that she was rough and impatient. I never wanted to see her again. This was 15 years ago but I have not forgotten. I have to agree with the other comment.  He doesnt listen to you.  He also does not really seem to care. Dr seemed disinterested in treating me. He had a poor bedside manner. I had a variety of concerns since I hadn't seen a doctor in quite sometime. I had many questions, which I had carefully listed so I wouldn't forget to ask. They were important to me. Every time I'd start to ask/explain my question/concern, she had cut me off about halfway through and just give me a short, sharp answer. I have had a lot of troubles with birth control and taken many different pills. When I tried to talk to her about this and choose a new type of pill, she gave me no time at all. Just shoved a perscription in my hand and rushed me out of there. I was only in there about 15 minutes. I guess she had somewhere to be (I was the last appt, and yes I was very early). She doesn't listen and gives horrible answers. I'll go to someone else with my concerns. $110 for 4 minutes of a Drs time.  The last two times I went, I had waited longer than an hour for my 4 minutes to get a RX renewed.Currently looking for someone closer. The Dr. I seen in her office acted like what I said didn't matter & that I knew nothing about my child. He has slight aniema & was given perscription vitimans with only 1/2 the stuff as the Flintstones Complete has ,& they have IRON in it . The ones they perscribed had  again only 1/2 the vitiams with NO IRON in them,BUT they had floride, for aniema!I told them I gave him floride treatments at home,they acted like I never said it. When I asked over & over , even to the staff {nurses}why there was no iron in the vitimans , I NEVER got answer. I asked the pharmisists about it & she gave me the RX bottle they came in . THAT'S how I found out there was no Iron in them . I asked a number of times after that too , & STILL NO ANSWER . I got totally ignored !! Does not spend enough time with me and does not listen well..Often in a hurry and overlooks complaints does not order the appropriate tests before rushing to a diagnosis..I went to this doctor with serious symptoms. He told me to go home and drink water. After a lot to questions that I had he finally agree to send me to a laboratory. Fortunately, I was right. Only after this he called me and sent a prescription for antibiotics to the pharmacy. was my son's pediatrician for 2 years. One night we called because our son became very I'll. He was very upset about beig bothered. The next day we brought him in to see him. Dr MIT hill couldn't look us in the eye. Dr Yu performed Mohs surgery on me a few years back.  I scheduled an appointment with him months age that was today at 1:50 to check out a few skin blemishes.  At 2:30 I was still waiting to get in and asked why I was waiting at 2:30 for a 1:50 appointment?  They told me he was behind but would see me soon.  I explained my time was important too and I left.  Dr Yu says he accepts new patients.  He already has too many and double/over books his schedule.  I will find a new Dermatologist and suggest others think twice before making Dr Yu your dermatologist unless you are willing to sit in a waiting room for hours. not a very nice bedside manner. "
myText3 = "Dr. Thakral takes the time to listen and thinks about the patient's concerns. I really like him, he spends time with you to explain things, asks if you have any questions, anf makes you feel like you are important. i was so scared, but he took very good care of me. The waiting room was almost empty when I arrived, and I filled out my new patient information on a digital tablet.  They have a concierge table, where a woman got me started on my paperwork.  Every staff member is very polite (huge), including Dr. Steuer.  Our wait time was around 45 minutes, which is very good.  The doctor's assistant seemed to care about me, and made a connection.  As opposed to my previous neurosurgeon, Dr. Steuer gave me much more information, and suggestions about what to do, even though his diagnosis was the same as my previous doctors.  He was honest with me, and said surgery was not recommended, both because I was very young, and mainly because it was not required. Dr. Phan is a typical Johns Hopkins physician- paying attention to the details of my health, humble, caring and outstanding. I saw him in 2010. He was excellent. I note those other comments are from another office years ago. I'm not sure some of the ratings would be accurate here although they were excellent. Some deducted points for office which isn't even the same anymore. You couldn't get nicer office than the one I visited. I have been seeing Dr. Davis for about 5 years now. I've been with her through 3 offices moves in the same building. She is awesome. She is always positive and just an easy doctor to talk to about any of my reproductive issues. Dr. Thakral takes the time to listen and thinks about the patient's concerns. I really like him, he spends time with you to explain things, asks if you have any questions, anf makes you feel like you are important. i was so scared, but he took very good care of me. The waiting room was almost empty when I arrived, and I filled out my new patient information on a digital tablet.  They have a concierge table, where a woman got me started on my paperwork.  Every staff member is very polite (huge), including Dr. Steuer.  Our wait time was around 45 minutes, which is very good.  The doctors assistant seemed to care about me, and made a connection.  As opposed to my previous neurosurgeon, Dr. Steuer gave me much more information, and suggestions about what to do, even though his diagnosis was the same as my previous doctors.  He was honest with me, and said surgery was not recommended, both because I was very young, and mainly because it was not required. Dr. Phan is a typical Johns Hopkins physician- paying attention to the details of my health, humble, caring and outstanding. I saw him in 2010. He was excellent. I note those other comments are from another office years ago. I am not sure some of the ratings would be accurate here although they were excellent. Some deducted points for office which isn't even the same anymore. You couldn't get nicer office than the one I visited. I have been seeing Dr. Davis for about 5 years now. I have been with her through 3 offices moves in the same building. She is awesome. She is always positive and just an easy doctor to talk to about any of my reproductive issues. He is very knowledgeable and he takes the time to explain everything. He has a great bedside manner. His staff is very friendly and helpful. He was easy to took to, Gave great advice. I am comfortable with my Choice. His Staff was very Sweet. Dr. Shearer has always taken time with my children and myself to properly diagnos problems.  Often he is able to reveal what the family Dr. cannot. Previous derms have backed away and looked down their noses at me.  She looks closely, gives good advice, listens, nice sense of humor.  I'm completely satisfied .. have had one biopsy, one freeze-off treatment. Dr. Brodsky is my personal physician, and is caring and compassionate. He take the time to listen to my problems and treats me with utmost care. DOCTOR WAS AWESOME! dr Johnson is fast efficient and thorough. Excellent Dr, knows what he's doing, great bedside manner and excellent excision skills. I was impressed at the attention she paid.  She made sure she answered all my questions and concerns.I had a concern and she saw me early and sent me for an ultrasound.  The doctor I was originally would not see me for another 3 weeks even though i told his office I had bleeding and was concerned about a miscarriage. Thoughtful, attentive and sees me for who I am and not just my condition. I appreciated the time she took to listen and not rush when I had questions or concerns. I am 19 years old and was diagnoised with a brain anursym in 2011. My mom was so worried on what was going to happen or what could have happened. We are from the flint, Michigan area and all nuerologist i have seen around here mad me feel as if i was just number, There was alot of life style changes i had to work on and seeing doctor Moskowitz as deffiantly made it an experence that i have been able to deal with. Dr Gerten was very personable and answered all my questions. I was impressed with the amount of time she spent with me. She is my 3 kids Doctor. she is very accurate and spends time with them. Dr. Barr is very good about answering questions & providing information to the patient. " 
words = word_tokenize(myText)
words1 = word_tokenize(myText1)
words2 = word_tokenize(myText2.lower())

def chunkWords(text):
    return(ne_chunk(pos_tag(word_tokenize(text))))
#print (chunkWords(myText));

tree = chunkWords(myText1);
#tree.pprint()
#tree.draw()

frequencies = Counter(words2)
#print (frequencies);

stemmedWordList = []
bigramsList = []


for i in words2:
    stemmedWord = stemmer.stem(i)
    stemmedWordList.append(stemmedWord)
print(stemmedWordList)   

stemmedFrequencies = Counter(stemmedWordList)
print (stemmedFrequencies);

#Bigrams
bigrams = list(ngrams(words2, 3))
bigramsList = bigramsList + bigrams

frequencies = Counter(bigramsList)
for i, count in frequencies.most_common(20):
    if("." in i or "," in i):
        continue
    print (i, count)
    
bigrams = list(ngrams(words2, 3))
bigramsList = bigramsList + bigrams

frequencies = Counter(bigramsList)
for i, count in frequencies.most_common(20):
    if("." in i or "," in i):
        continue
    print (i, count)
