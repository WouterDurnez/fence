SYSTEM_PROMPT = '''You are an instructor in charge of creating exam questions.

You will be given a piece of text, and I would like you to create an exam question based on it.

The exam question is open ended. It is followed by multiple choice answer options with four options.

The question and answers should be written in {language_name}.

The output should be in a single line TOML format enclosed by triple backticks, similar to the following example:

```toml
question = """*your generated question, which is fully self-contained and has no references to the text*"""

[[answers]]
text = """response option 1"""
correct = true
reason = """*explain briefly why this is the correct response here*"""

[[answers]]
text = """response option 2"""
correct = false
reason = """*explain briefly why this is not the correct response here*"""

[[answers]]
text = """response option 3"""
correct = false
reason = """*explain briefly why this is not the correct response here*"""

[[answers]]
text = """response option 4"""
correct = false
reason = """*explain briefly why this is not the correct response here*"""
```

Guidelines:
- You are ONLY allowed to add valid TOML wrapped in triple backticks.
- Make sure to mark the correct response in the TOML using booleans. Only one answer can be correct.
- Always add valid response options: "Don't know" is NEVER a valid response. You WILL be penalized for including this in the response options.
- You must wrap all values in triple quotes, even if they are single words or numbers. This helps to ensure that the output is valid TOML, and that the parser can handle the output.
- The generated question and answers MUST BE in the {language_name} language.
'''

# INPUT

USER_MESSAGE = """
This is the text:

```{highlight}```
"""

HIGHLIGHT = """
Mixed martial arts (MMA)[a] is a full-contact fighting sport based on striking and grappling, incorporating techniques from various combat sports from around the world.[10]

In the early 20th century, various inter-stylistic contests took place throughout Japan and the countries of East Asia. At the same time, in Brazil there was a phenomenon called vale tudo, which became known for unrestricted fights between various styles such as judo, Brazilian jiu-jitsu, catch wrestling, luta livre, Muay Thai and capoeira. An early high-profile mixed bout was Kimura vs. Gracie in 1951. In mid-20th century Hong Kong, rooftop street fighting contests between different martial arts styles gave rise to Bruce Lee's hybrid martial arts style Jeet Kune Do. Another precursor to modern MMA was the 1976 Ali vs. Inoki exhibition bout, fought between boxer Muhammad Ali and wrestler Antonio Inoki in Japan, where it later inspired the foundation of Shooto in 1985, Pancrase in 1993, and the Pride Fighting Championships in 1997.

In the 1990s, the Gracie family brought their Brazilian jiu-jitsu style, first developed in Brazil from the 1920s, to the United States—which culminated in the founding of the Ultimate Fighting Championship (UFC) promotion company in 1993. The company held an event with almost no rules, mostly due to the influence of Art Davie and Rorion Gracie attempting to replicate mixed contests that existed in Brazil[11] and Japan. They would later implement a different set of rules (example: eliminating kicking a grounded opponent), which differed from other leagues which were more in favour of realistic, "street-like" fights.[12] The first documented use of the term mixed martial arts was in a review of UFC 1 by television critic Howard Rosenberg in 1993.

Originally promoted as a competition to find the most effective martial arts for real unarmed combat, competitors from different fighting styles were pitted against one another in contests with relatively few rules.[13] Later, individual fighters incorporated multiple martial arts into their style. MMA promoters were pressured to adopt additional rules to increase competitors' safety, to comply with sport regulations and to broaden mainstream acceptance of the sport.[14] Following these changes, the sport has seen increased popularity with a pay-per-view business that rivals boxing and professional wrestling.[15]

History
Antiquity

A Chinese martial artist preparing to throw his opponent during a lei tai contest in ancient China
In ancient China, combat sport appeared in the form of Leitai, a no-holds-barred mixed combat sport that combined Chinese martial arts, boxing and wrestling.[16]


The Pancrastinae: a statue portraying the pancratium, an event which took place in the Roman Colosseum. Even as late as the Early Middle Ages, statues were put up in Rome and other cities to honor remarkable pankratiasts. This statue, now part of the Uffizi collection, is a Roman copy of a lost Greek original, circa 3rd century BC.

Ancient Greek pankratiasts fighting. This drawing is an early 20th century copy of a scene from a Panathenaic amphora.[17]
In ancient Greece, there was a sport called pankration, which featured grappling and striking skills similar to those found in modern MMA. Pankration was formed by combining the already established wrestling and boxing traditions and, in Olympic terms, first featured in the 33rd Olympiad in 648 BC. All strikes and holds were allowed with the exception of biting and gouging, which were banned. The fighters, called pankratiasts, fought until someone could not continue or signaled submission by raising their index finger; there were no rounds.[18][19] According to the historian E. Norman Gardiner, "No branch of athletics was more popular than the pankration."[20] There is also evidence of similar mixed combat sports in ancient Egypt, India and Japan.[16]

Modern-era precursors
The mid-19th century saw the prominence of the new sport savate in the combat sports circle. French savate fighters wanted to test their techniques against the traditional combat styles of its time. In 1852, a contest was held in France between French savateurs and English bare-knuckle boxers in which French fighter Rambaud alias la Resistance fought English fighter Dickinson and won using his kicks. However, the English team still won the four other match-ups during the contest.[21] Contests occurred in the late 19th to mid-20th century between French savateurs and other combat styles. Examples include a 1905 fight between French savateur George Dubois and a judo practitioner Re-nierand which resulted in the latter winning by submission, as well as the highly publicized 1957 fight between French savateur and professional boxer Jacques Cayron and a young Japanese karateka named Mochizuki Hiroo which ended when Cayron knocked Hiroo out with a hook.[21]

Catch wrestling appeared in the late 19th century, combining several global styles of wrestling, including Indian pehlwani and English wrestling.[22][23] In turn, catch wrestling went on to greatly influence modern MMA.[citation needed][24] No-holds-barred fighting reportedly took place in the late 1880s when wrestlers representing the style of catch wrestling and many others met in tournaments and music-hall challenge matches throughout Europe. In the US, the first major encounter between a boxer and a wrestler in modern times took place in 1887 when John L. Sullivan, then heavyweight world boxing champion, entered the ring with his trainer, wrestling champion William Muldoon, and was slammed to the mat in two minutes. The next publicized encounter occurred in the late 1890s when future heavyweight boxing champion Bob Fitzsimmons took on European wrestling champion Ernest Roeber. In September 1901, Frank "Paddy" Slavin, who had been a contender for Sullivan's boxing title, knocked out future world wrestling champion Frank Gotch in Dawson City, Canada.[25] The judo-practitioner Ren-nierand, who gained fame after defeating George Dubois, would fight again in another similar contest, which he lost to Ukrainian Catch wrestler Ivan Poddubny.[21]

Another early example of mixed martial arts was Bartitsu, which Edward William Barton-Wright founded in London in 1899. Combining catch wrestling, judo, boxing, savate, jujutsu and canne de combat (French stick fighting), Bartitsu was the first martial art known to have combined Asian and European fighting styles,[26] and which saw MMA-style contests throughout England, pitting European catch wrestlers and Japanese judoka champions against representatives of various European wrestling styles.[26]

Among the precursors of modern MMA are mixed style contests throughout Europe, Japan, and the Pacific Rim during the early 1900s.[27] In Japan, these contests were known as merikan, from the Japanese slang for "American [fighting]". Merikan contests were fought under a variety of rules, including points decision, best of three throws or knockdowns, and victory via knockout or submission.[28]

Sambo, a martial art and combat sport developed in Russia in the early 1920s, merged various forms of combat styles such as wrestling, judo and striking into one unique martial art.[29][30] The popularity of professional wrestling, which was contested under various catch wrestling rules at the time, waned after World War I, when the sport split into two genres: "shoot", in which the fighters actually competed, and "show", which evolved into modern professional wrestling.[31] In 1936, heavyweight boxing contender Kingfish Levinsky and professional wrestler Ray Steele competed in a mixed match, which catch wrestler Steele won in 35 seconds.[31] 27 years later, Ray Steele's protégé Lou Thesz fought boxer Jersey Joe Walcott twice in mixed style bouts. The first match was a real contest which Thesz won while the second match was a work, which Thesz also won.

In the 1940s in the Palama Settlement in Hawaii, five martial arts masters, under the leadership of Adriano Emperado, curious to determine which martial art was best, began testing each other in their respective arts of kenpo, jujitsu, Chinese and American boxing and tang soo do. From this they developed kajukenbo, the first American mixed martial arts.


Masahiko Kimura vs. Hélio Gracie, a 1951 bout between Japanese judo fighter Masahiko Kimura and Brazilian jiu jitsu founder Hélio Gracie in Brazil, was an early high-profile mixed martial arts bout.
In 1951, a high-profile grappling match was Masahiko Kimura vs. Hélio Gracie, which was wrestled between judoka Masahiko Kimura and Brazilian jiu jitsu founder Hélio Gracie in Brazil. Kimura defeated Gracie using a gyaku-ude-garami armlock, which later became known as the "Kimura" in Brazilian jiu jitsu.[32] In 1963, a catch wrestler and judoka "Judo" Gene Lebell fought professional boxer Milo Savage in a no-holds-barred match. Lebell won by Harai Goshi to rear naked choke, leaving Savage unconscious. This was the first televised bout of mixed-style fighting in North America. The hometown crowd was so enraged that they began to boo and throw chairs at Lebell.[33]

On February 12, 1963, three karatekas from Oyama dojo (kyokushin later) went to the Lumpinee Boxing Stadium in Thailand and fought against three Muay Thai fighters. The three kyokushin karate fighters were Tadashi Nakamura, Kenji Kurosaki and AkiFujihira (also known as Noboru Osawa), while the Muay Thai team of three authentic Thai fighter.[34] Japan won 2–1: Tadashi Nakamura and Akio Fujihira both knocked out their opponents with punches while Kenji Kurosaki, who fought the Thai, was knocked out by elbows. The Japanese fighter who lost, Kenji Kurosaki, was a kyokushin instructor, rather than a contender, and that he had stood in as a substitute for the absent chosen fighter. In June of the same year, karateka and future kickboxer Tadashi Sawamura faced top Thai fighter Samarn Sor Adisorn: Sawamura was knocked down sixteen times on his way to defeat.[34] Sawamura went on to incorporate what he learned in that fight in kickboxing tournaments.


Bruce Lee popularized the concept of mixed martial arts via his hybrid philosophy of Jeet Kune Do during the late 1960s to early 1970s.
During the late 1960s to early 1970s, the concept of hybrid martial arts was popularized in the West by Bruce Lee via his system of Jeet Kune Do.[35] Lee believed that "the best fighter is not a boxer, karate or judo man. The best fighter is someone who can adapt to any style, to be formless, to adopt an individual's own style and not following the system of styles."[36] In 2004, UFC President Dana White would call Lee the "father of mixed martial arts" stating: "If you look at the way Bruce Lee trained, the way he fought, and many of the things he wrote, he said the perfect style was no style. You take a little something from everything. You take the good things from every different discipline, use what works, and you throw the rest away".[37]

A contemporary of Bruce Lee, Wing Chun practitioner Wong Shun Leung, gained prominence fighting in 60–100 illegal beimo fights against other Chinese martial artists of various styles. Wong also fought and won against Western fighters of other combat styles, such as his match against Russian boxer Giko,[38] his televised fight against a fencer,[39] and his fight against Taiwanese kung fu master Wu Ming Jeet.[40] Wong combined boxing and kickboxing into his kung fu, as Bruce Lee did.


Muhammad Ali vs. Antonio Inoki, a 1976 bout in Japan where boxer Muhammad Ali fought wrestler Antonio Inoki, was an important precursor to MMA contests.
Muhammad Ali vs. Antonio Inoki took place in Japan in 1976. The classic match-up between professional boxer and professional wrestler turned sour as each fighter refused to engage in the other's style, and after a 15-round stalemate it was declared a draw. Muhammad Ali sustained a substantial amount of damage to his legs, as Antonio Inoki slide-kicked him continuously for the duration of the bout, causing him to be hospitalized for the next three days.[41] The fight played an important role in the history of mixed martial arts.[42]

The basis of modern mixed martial arts in Japan can be found across several shoot-style professional wrestling promotions such as UWF International and Pro Wrestling Fujiwara Gumi, both founded in 1991, that attempted to create a combat-based style which blended wrestling, kickboxing and submission grappling. Another promotion formed around the same time by Akira Maeda called Fighting Network RINGS initially started as a shoot-style professional wrestling promotion but it also promoted early mixed martial arts contests. From 1995 onwards it began identifying itself as a mixed martial arts promotion and moved away from the original shoot style. Professional wrestlers Masakatsu Funaki and Minoru Suzuki founded Pancrase in 1993 which promoted legitimate contests initially under professional wrestling rules. These promotions inspired Pride Fighting Championships which started in 1997. Pride was acquired by its rival Ultimate Fighting Championship in 2007.[43][44]

A fight between Golden Gloves boxing champion Joey Hadley and Arkansas Karate Champion David Valovich happened on June 22, 1976, at Memphis Blues Baseball Park. The bout had mixed rules: the karateka was allowed to use his fists, feet and knees, while the boxer could only use his fists. Hadley won the fight via knockout on the first round.[45]

In 1988 Rick Roufus challenged Changpuek Kiatsongrit to a non-title Muay Thai vs. kickboxing super fight. Roufus was at the time an undefeated Kickboxer and held both the KICK Super Middleweight World title and the PKC Middleweight U.S. title. Kiatsongrit was finding it increasingly difficult to get fights in Thailand as his weight (70 kg) was not typical for Thailand, where competitive bouts tended to be at the lower weights. Roufus knocked Changpuek down twice with punches in the first round, breaking Changpuek's jaw, but lost by technical knockout in the fourth round due to the culmination of low kicks to the legs that he was unprepared for. This match was the first popular fight which showcased the power of such low kicks to a predominantly Western audience.[46]"""

LANGUAGE_NAME = "english"
