# A Plain-English Guide to the AI Yoga Posture Correction Research Paper

*This guide breaks down the core concepts, technical innovations, and key takeaways of our IEEE academic research paper into simple, easy-to-understand language. It’s perfect for preparing for a presentation or explaining the project to non-technical stakeholders.*

---

## 1. What is the Big Problem Being Solved?
Since 2020, people have flocked to online video platforms (like YouTube and Zoom) for yoga and fitness. The biggest issue? **There is no two-way feedback.** 
When you practice in an actual studio, a teacher walks around and physically adjusts your pose. At home, you might be doing a pose incorrectly for months, which leads to joint strain and injury.

Our paper solves this by introducing a **"Desktop AI Helper."** It watches the online video, watches you through your webcam, and acts as your personal, automated instructor.

## 2. The Core Innovation: "The Ghost Skeleton"
Most AI yoga apps make you look away from your mat to stare at a tiny computer screen to check if your body matches a graph. This breaks your focus and balance!

We invented the **Ghost Skeleton overlay**:
1. The AI looks at the instructor on the screen and draws an invisible "stick figure" over them.
2. It then looks at you through the webcam.
3. **The Magic:** It dynamically scales the instructor's skeleton to perfectly match your body height and shoulder width. 
4. It projects this "Ghost Skeleton" right onto your video feed like a chalk outline. 

### Why the Colors Matter
Instead of guessing what you are doing wrong, the skeleton's limbs change color based on mathematical angles:
- 🟩 **Green:** You match the instructor perfectly!
- 🟦 **Cyan:** You are close, but need slight adjusting.
- 🟧 **Orange:** You are significantly out of alignment.
- 🟥 **Red:** Critical misalignment — fix this immediately to avoid injury.

## 3. Real-Time Multilingual Audio (No Neck-Craning!)
Because looking at a screen during yoga is hard, the system talks to you. 
If the AI detects that your elbow angle is too sharp compared to the instructor's, it builds a sentence like: *"Straighten your elbow."*

To make this globally accessible, we tied it to a translation engine. It can instantly translate the feedback into 14 different languages (like Spanish, Hindi, or French) so anyone around the world can use it.

## 4. Why We Didn't Use Expensive Hardware
A lot of complex AI requires massive gaming graphics cards (GPUs) that everyday people don't own. 

Our paper details how we bypassed this:
- We used **Google MediaPipe**, a super-lightweight system that tracks 33 points on the human body.
- We purposefully **slowed the AI down**. Since yoga is slow, we don't need the camera to check you 60 times a second. We set it to check you **5 times a second**. 
- Because of this clever optimization, the app runs perfectly on a standard work laptop without overheating or causing the Zoom call to crash!

## 5. What Were the Results of the Experiment?
We tested the system on 15 amateur yoga practitioners. 
- **High Accuracy:** The AI correctly flagged bodily misalignments **92.1% of the time**. It is incredibly accurate on standing poses like "Warrior II" and "Tree Pose."
- **User Satisfaction:** Users rated the audio-feedback exceptionally high (4.8 out of 5), reporting that it saved them from having to awkwardly twist their necks to look at their screens.

## 6. What Are the Limitations? 
The research acknowledges a few weaknesses:
- **Baggy Clothes:** If someone wears huge sweatpants, the AI struggles to guess exactly where their knee is under the fabric.
- **Facing the Camera:** If a user does a downward fold where the top of their head is pointed directly into the camera lens, the AI loses its depth perception. We fixed this by having the AI politely ask the user to turn sideways.

## 7. Next Steps for the Project
In the "Future Work" section, the paper notes goals like:
- **Action Tracking:** Upgrading the AI to grade the *transitions* between poses, not just the static pose itself.
- **Multi-Person Classes:** Right now, if there are 5 people in the YouTube video, the AI might get confused. We plan to add a tool that completely ignores everyone except the main instructor.

---
**Summary for Presentations:**
*This project isn't just about cool computer vision; it's about making fitness safer and more accessible. By combining visual ghost overlays with translated audio cues running efficiently on standard laptops, we’ve created a practical, real-world utility that brings the studio experience directly into the living room.*
