This repository contains the codebase for evaluating and ranking mentors for the WnCC Seasons of Code (SoC), based on student progress, mentor responsiveness, engagement depth, and mentee feedback.

Project Structure->
main.py                 # Core logic and execution script
requirements.txt        # Python dependencies
mentors.csv             # Input data (Mentor details)
students.csv            # Input data (Mentee progress)
interactions.csv        # Input data (Interactions & response times)
Ideation_Document.txt   # Mathematical justifications
README.md               # This documentation file

SetUp Instructions->
1) Install python lastest version
2) Install Numpy and Pandas Libraries

Run the program-->
To run the program type "python main.py" in the dedicated terminal, ensuring the structure of the project is same as above mentioned

<-----IDEATION DOCUMENT------->

1. Responsiveness Score R(t_avg)

Requirement: Design a bounded function [0, 1] where faster responses yield higher scores and slow responses are heavily penalized.

Chosen Functional Form:
I utilized a Rational Decay Function (inverse square):
R(t_avg) = 1 / (1 + (t_avg / t_half)^2)

Justification:
* t_half is set to 24 hours, representing the "half-life" of a good response. A mentor averaging 24 hours receives a balanced score of 0.5.
* The Exponent (2) ensures the penalty curve is aggressive for extreme delays. 
* Unlike a linear function, this never yields negative numbers and asymptotes cleanly to 0 for heavily delayed responses (e.g., a 72-hour average drops the score to 0.10), while highly rewarding rapid replies (e.g., a 4-hour average yields ~0.97).


2. Component Weight Justification

Requirement: Justify the weights w1, w2, w3, w4 such that they sum to 1.

Chosen Weights & Justification:
* w1 = 0.40 (Student Progress - P): The primary objective of WnCC's SoC is project completion. If mentees are not completing milestones, the mentorship is functionally falling short, making this the heaviest weighted indicator.
* w3 = 0.30 (Engagement - E): Deep technical engagement (Code Reviews, Meetings) is the core mechanism of mentorship. It receives the second-highest weight as it measures the tangible effort and time the mentor is investing.
* w2 = 0.15 (Responsiveness - R): While replying quickly is helpful, a delayed but thorough code review is vastly superior to a lightning-fast but unhelpful message. Thus, pure speed is weighted lower than actual engagement.
* w4 = 0.15 (Mentee Feedback - F): Student feedback is inherently subjective. Even with mathematical smoothing, students may rate a mentor poorly simply because a project is difficult. Keeping this weight at 15% respects student voices while preventing the score from becoming a pure popularity contest.

3. Handling Mentee Feedback Bias
---
Requirement: Design a method to detect and down-weight unreliable or extremely biased ratings.

Chosen Mechanism:
Instead of a simple arithmetic mean, the system applies a Bayesian Average to normalize the feedback score (F):

F_raw = (v / (v + m)) * x_bar  +  (m / (v + m)) * C

* v: The number of reviews for the specific mentor.
* m: A confidence threshold (set to 3). This is the minimum number of reviews required before we start heavily trusting the mentor's individual average.
* x_bar: The mentor's arithmetic mean rating.
* C: The global mean of all mentor ratings across the SoC program.

Justification:
If a mentor has only 1 mentee who leaves a malicious "1-star" rating, the low sample size (v=1) gives more weight to the global average (C). This pulls the unfair 1-star rating up toward the program average (e.g., yielding a 3.4), effectively dampening the outlier. As a mentor accumulates more legitimate reviews, v grows, and their true average (x_bar) takes over the equation.

4. Key Assumptions

1. Feedback Data Availability: The provided CSV schemas did not explicitly include a 1-5 rating column. The codebase assumes the existence of a standard feedback dataset to compute F. If absent, the system gracefully defaults to a neutral score of 0.5 for all mentors to ensure the pipeline does not break.
2. Internal Engagement Weights: It is assumed that not all interactions require equal effort. When calculating the raw Engagement Score, the system internally weights Code Reviews (5 points) and Meetings (3 points) significantly higher than standard Messages (0.1 points).
3. Cross-Project Normalization: For mentors managing multiple projects, it is assumed that their total bandwidth is divided among their total distinct mentees. Engagement is normalized by dividing the weighted interaction sum by the distinct count of Student IDs mapped to that mentor, preventing unfair advantages for mentors with large cohorts.

