import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pprint import pprint

from fence import Chain, ClaudeInstant
from utils import build_links
from fence.utils.logger import setup_logging

logger = setup_logging(__name__)

claude_model = ClaudeInstant(source="test-faq")


def handler(event, context):
    logger.info("👋 Let's rock!")

    # Parse event
    input_text = event.get("input_text", {})
    logger.info(f"📥 Received input: {input_text}")

    # Create links and chain
    links = build_links(llm=claude_model)
    chain = Chain(links=links, llm=claude_model)

    # Run chain
    chain_output = chain.run(input_dict={"highlight": input_text})

    # Build response
    return {"statusCode": 200, "body": chain_output}


if __name__ == "__main__":
    # Text snippet
    snippet = """Telecommunication with photorealistic avatars in virtual or augmented reality is a promising path for achieving authentic face-to-face communication in 3D over remote physical distances. In this work, we present the Pixel Codec Avatars (PiCA): a deep generative model of 3D human faces that achieves state of the art reconstruction performance while being computationally efficient and adaptive to the rendering conditions during execution. Our model combines two core ideas: (1) a fully convolutional architecture for decoding spatially varying features, and (2) a rendering-adaptive per-pixel decoder. Both techniques are integrated via a dense surface representation that is learned in a weakly-supervised manner from low-topology mesh tracking over training images. We demonstrate that PiCA improves reconstruction over existing techniques across testing expressions and views on persons of different gender and skin tone. Importantly, we show that the PiCA model is much smaller than the state-of-art baseline model, and makes multi-person telecommunication possible: on a single Oculus Quest 2 mobile VR headset, 5 avatars are rendered in realtime in the same scene.    """

    # Another snippet
    academic_snippets = [
        "This work contributes towards a deeper understanding of synchrony by focusing on triads, using VR to simulate avatars based in two different environments, and testing two different kinds of design tasks. Specifically, our contributions are (1) a study protocol for co-located VR studies using triads, (2) new methods for measuring triadic synchrony, and exploration of the properties of each method, (3) confirmatory results that synchrony occurs between teammates using VR, (4) confirmatory results that synchrony occurs among triads, (5) evidence that the context of virtual environment can affect synchrony, (6) evidence that speaking roles affect synchrony, and (7) evidence that gaze influences synchrony.",
        "The entity processing task is critical in natural language processing, and its success or failure directly affects the per- formance of natural language processing [75]. At present, there are two main methods for entity extraction, one is to formulate regular expressions for automatic extraction, and the other is the entity mentioned in the manual tagged doc- ument. However, neither of these two methods can extract entities efficiently and accurately. Zhang et al. [56] propose a human-in-the-loop-based entity extraction method to obtain the best return on investment in a limited time. It mainly commands humans to formulate regular expressions and mark documents. The whole pipeline contains three steps. First, they use regular expressions to scan the document corpus and generate weak labels to pre-train the neural network. Then, they manually annotate substring to fine-tune the network and use this fine-tuned network to identify continuously. Finally, they complete an entity extraction model suitable for tasks in the professional field. Besides, this regular expression model can also be constantly upgraded and trained to achieve an ef- ficient and accurate general recognition effect. With the deep- ening of research, we need to deal with more and more tasks. The emergence of new schemes is beyond our expectations. Regular expressions can help handle common data, but there is no expected magic for new data never seen before. To tackle this challenge, some studies proposed approaches to solve the cross-domain problem in entity links. They find the entities mentioned in the text and filter and discriminate them according to entities sorting information. This method is es- pecially suitable for semantic disambiguation tasks [64].",
        "Recently, deep learning models, particularly convolutional neural networks (CNN), surpassed traditional computer vision methods for semantic segmentation. In contrast to the conventional approach based on hand-crafted features, CNNs are able to automatically learn high-level fea- tures adapted specifically to the task of brain tumor segmentation. Currently, the vast majority of effective CNNs for medical image segmentation are based on a U-Net [23] architecture with millions of trainable parameters. However, such complex models can be highly prone to over- fitting especially in cases where the amount of training data is insufficient, which is usually the case for medical imaging. In this work, we introduce a new layer, called low-rank convolution, in which low-rank constraints are imposed to regularize weights and thus reduce overfitting. We make use of a 3D U-Net [5] architecture with residual modules [10] and further improve it by replacing ordinary convolution layers with low-rank ones, achieving models with several times fewer parameters than the initial ones. This leads to significantly better performance especially because the amount of training data is limited.",
        "Affective Computing has emerged to fulfill this gap by converging technology and emotions into HCI. It aims to model emotional interactions between a human and a computer by measuring the emotional state of a user [2]. A persons inner emotional state may become apparent by subjective experiences (how the person feels), internal/in- ward expressions (physiological signals), and external/out- ward expressions (audio/visual signals) [3]. Subjective self- reports about how the person is feeling can provide valuable information but there are issues with validity and corrobo- ration [4]. Participants may not answer exactly how they are feeling but rather as they feel others would answer. Physiological signals can assist in obtaining a better understanding of the participants underlying responses expressed at the time of the observations. These correspond to multichannel recordings from both the central and the au- tonomic nervous systems. The central nervous system com- prises the brain and spinal cord, while the autonomic ner- vous system is a control system that acts unconsciously and regulates bodily functions such as the heart rate, pupillary response, and sexual arousal. The signals commonly used to measure emotions are the Galvanic Skin Response (GSR), which increases linearly with a persons level of arousal; Electromyography (EMG) (frequency of muscle tension), which is correlated with negatively valenced emotions; Heart Rate (HR), which increases with negatively valenced emotions such as fear; and Respiration Rate (RR) (how deep and fast the breath is), which becomes irregular with more aroused emotions like anger. Measurements recorded over the brain also enable the observation of the emotions felt [3].",
        "In an attempt to address this shortcoming, Jia, Shi, Zang and Mu ̈ller (2013) conducted a series of bimodal TOJ experiments, in which participants judged the order of audio-tactile stimulus pairs. Results showed that the prior presentation of affectively salient pictures—at a location independent of the audio-tactile stimuli—was capable of shifting attention towards the somatosensory modality, resulting in the quicker perception of tactile stimuli com- pared to concomitant auditory stimuli. Notably, this effect was only found when stimuli from different modalities were also separated in space. Prioritization effects were found for both positive (e.g., an erotic couple) and negative (e.g., a spider) high-arousal imagery. When disentangling the effects of physically threatening contexts with regard to the locus of threat, prioritization of somatosensory stimuli only occurred when the visual cue represented a near-body threat (e.g., a snake), and not when it depicted remote threat (e.g., a car accident). A limitation of the aforemen- tioned studies (Jia et al., 2013; Van Damme et al., 2009) is that only visual threat cues were used. Effects of the actual anticipation of pain thus remain open to investigation.",
        "TO TELL YOU the truth, it was a long time ago and I didn't really spend much time with them. It's pushing it to say I knew them at all, really. It was just a job, you see. A well paid commission between my more serious works. I never expected it to become... well, the thing I am most famous for. I doubt any of them remember me. I honestly doubt any of them are alive any more. It's been sixty years since the hive war on Verghast, and Imperial Guardsman is not a career with long-term prospects. No, they're probably all long dead by now. If so, may the Emperor of Mankind rest them, every one. I had a friend who worked in the Munitorium at NorthCol who was kind enough to pass me copies of Imperial dispatches so I could follow their movements and fortunes. For a few years, it pleased me to keep track of them. When I read of their successes on Hagia and Phantine, I poured a glass of joiliq and sat in my studio, toasting their name.",
    ]

    for snippet in academic_snippets[:3]:
        # Call the handler
        response = handler({"input_text": snippet}, None)

        # Print response
        pprint(response["body"]["clean_question"])
