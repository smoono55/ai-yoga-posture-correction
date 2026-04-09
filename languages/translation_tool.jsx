import { useState, useRef } from "react";

const LANGS = [
  { code: "es", name: "Spanish",    tts: "es-ES-ElviraNeural" },
  { code: "fr", name: "French",     tts: "fr-FR-DeniseNeural" },
  { code: "de", name: "German",     tts: "de-DE-KatjaNeural" },
  { code: "ja", name: "Japanese",   tts: "ja-JP-NanamiNeural" },
  { code: "zh", name: "Chinese",    tts: "zh-CN-XiaoxiaoNeural" },
  { code: "ar", name: "Arabic",     tts: "ar-EG-SalmaNeural" },
  { code: "pt", name: "Portuguese", tts: "pt-BR-FranciscaNeural" },
  { code: "ru", name: "Russian",    tts: "ru-RU-SvetlanaNeural" },
  { code: "ko", name: "Korean",     tts: "ko-KR-SunHiNeural" },
  { code: "it", name: "Italian",    tts: "it-IT-ElsaNeural" },
];

const EN_STRINGS = {"startup_greeting":"Yoga posture correction system is starting.","cmd_score":"{score} percent — {label}.","cmd_what_pose_known":"You're working on {pose_name}, or {sanskrit}.","cmd_what_pose_unknown":"I haven't identified the pose yet — give me a moment.","cmd_repeat_nothing":"Nothing to repeat yet.","cmd_pause":"Corrections paused. Say resume whenever you're ready.","cmd_resume":"Resuming. Let's go!","cmd_restart":"Restarting the guide for {pose_name} from the top.","cmd_help":"You can say: score, what pose, next step, repeat, pause, resume, or restart.","cam_too_close":"You're too close! Please step back so your full body fits in the frame.","cam_too_far":"You're a little far. Please move closer so I can see your full body.","cam_not_centred":"Move toward the centre of the frame.","cam_sideways":"It looks like you might be turned sideways. Please face the camera directly.","cam_ok":"Camera alignment looks good.","cam_low_visibility":"Some body parts aren't visible — check your lighting and make sure your full body is in frame.","pose_new_intro_prefix":"The instructor has moved into","pose_new_intro_prefix2":"We're going into {pose_name} now","pose_new_intro_prefix3":"Coming up is {pose_name}","pose_new_intro_prefix4":"Next pose is {pose_name}","event_hold_achieved":"Excellent! You've held {pose_name} successfully. Keep breathing and stay in the pose.","event_catching_up":"Nicely done holding that pose! Now let's catch up — the instructor is in {new_pose}. I'll guide you in.","event_frozen":"The instructor has moved on, but that's okay — let's focus on holding {pose_name} first. Get your score above {threshold} percent and hold it for {hold_sec} seconds.","score_excellent_1":"You're at {score} percent — excellent.","score_excellent_2":"{score} percent match. Beautiful.","score_excellent_3":"Ninety plus — your body is right where it needs to be.","score_good_1":"You're at {score} percent — good alignment, small adjustments left.","score_good_2":"{score} percent. Getting really close.","score_good_3":"Looking good at {score} — just a little more to refine.","score_fair_1":"{score} percent — you're making progress.","score_fair_2":"At {score} percent — keep working through the corrections.","score_fair_3":"{score} percent. Stay focused and keep breathing.","score_needs_work_1":"{score} percent right now — let's work through this together.","score_needs_work_2":"You're at {score} percent. No rush — keep adjusting.","score_needs_work_3":"{score} percent — take it one joint at a time.","well_done_1":"That's it — hold that and breathe.","well_done_2":"Yes! That's the shape. Stay with it.","well_done_3":"Beautiful alignment. Keep breathing.","well_done_4":"Perfect — you've found it. Don't move.","well_done_5":"That's exactly right. Nice work.","well_done_6":"Great shape. Lock it in.","well_done_7":"You're matching the instructor well. Maintain it.","well_done_8":"Spot on. Feel that alignment.","enc_small_1":"Almost there.","enc_small_2":"Really close now.","enc_small_3":"Nearly perfect.","enc_small_4":"Tiny adjustment.","enc_small_5":"You're so close.","enc_medium_1":"You're doing well — keep adjusting.","enc_medium_2":"Good effort — stay with it.","enc_medium_3":"Nice work — keep breathing through it.","enc_large_1":"Take your time — this one takes practice.","enc_large_2":"Breathe and keep working through it.","enc_large_3":"Every body is different — do what you can.","summary_no_attempts":"Your session is complete. I didn't record any full pose attempts this time. Great effort anyway!","summary_opening_with_time":"Great session! You practised for about {minutes} {minute_word}.","summary_opening_short":"Great effort on your session!","summary_minute":"minute","summary_minutes":"minutes","summary_all_held":"You successfully held every pose — {names}. That's a fantastic result!","summary_some_held":"You successfully held {held} out of {total} poses: {names}.","summary_missed_singular":"The pose to keep working on is {names}. Your best score was: {scores}. You'll get there with a little more practice!","summary_missed_plural":"The poses to keep working on are {names}. Your best scores were: {scores}. You'll get there with a little more practice!","summary_closer_1":"Remember, every practice counts — see you on the mat again soon!","summary_closer_2":"Consistency is the key to progress. Keep showing up!","summary_closer_3":"Each session makes you stronger. Well done today!","summary_closer_4":"Take a moment to breathe and celebrate your effort today.","pose_mountain":"Mountain Pose","pose_forward_fold":"Forward Fold","pose_halfway_lift":"Halfway Lift","pose_warrior_i":"Warrior One","pose_warrior_ii":"Warrior Two","pose_warrior_iii":"Warrior Three","pose_chair":"Chair Pose","pose_tree":"Tree Pose","pose_triangle":"Triangle Pose","pose_downward_dog":"Downward Facing Dog","pose_plank":"Plank Pose","pose_low_lunge":"Low Lunge","pose_high_lunge":"High Lunge","pose_cobra":"Cobra Pose","pose_upward_dog":"Upward Facing Dog","pose_child":"Child's Pose","pose_seated_forward":"Seated Forward Fold","pose_cat_cow":"Cat Cow","pose_unknown":"Unknown Pose","cam_cant_see":"I can't see you. Please stand in front of the camera in a well-lit area.","cam_parts_missing":"Parts of your body aren't visible. Please step back so your full body is in frame.","hold_grace_period":"The instructor has moved on, but don't worry — take your time to get into {pose_name}. You have about {breaths} more breath cycles.","hold_release":"No problem — let's move on with the instructor. We'll practise {pose_name} again next time."};
const EN_CORRECTIONS = {"correction_0":"Bend your left knee deeper — sink your weight into it.","correction_1":"Left knee is too straight — soften it and let your hips drop.","correction_2":"Your left leg needs more bend — track that knee over your toes.","correction_3":"Warrior Two lives in the legs — sink that left knee directly over your ankle.","correction_4":"Left knee wants to go deeper — this is where your strength lives.","correction_5":"Sit into that left knee — imagine lowering into a chair.","correction_6":"Left knee just a touch deeper — breathe and let gravity help.","correction_7":"Drop your left knee lower — press the floor away with your heel.","correction_8":"In this lunge, the front knee tracks right over your ankle — go deeper.","correction_9":"Straighten your left leg — press the heel firmly into the floor.","correction_10":"Left knee is bent too far — draw back and stack the knee over the heel.","correction_11":"Extend your left leg — engage the thigh and lengthen through the knee.","correction_12":"In Downward Dog the back leg is long — press your left heel toward the mat.","correction_13":"Left leg needs more length — activate the quad and press out.","correction_14":"Micro-straighten the left knee — just ease the bend slightly.","correction_15":"Bend your right knee deeper — let your hips descend.","correction_16":"Your right leg is too straight — soften the knee and sit lower.","correction_17":"Right knee needs more bend — track it forward over your middle toe.","correction_18":"Sink into your right knee — this is the powerhouse of this pose.","correction_19":"Right knee just a little deeper — use your breath to soften into it.","correction_20":"Front knee bends to ninety — keep pressing that right knee forward.","correction_21":"Straighten your right leg — activate the thigh muscles.","correction_22":"Right knee is over-bent — ease back and stack it over your foot.","correction_23":"Press your right heel down and let the leg lengthen naturally.","correction_24":"Right leg wants more extension — imagine the knee opening up.","correction_25":"That back leg is your anchor — press the right heel and lengthen.","correction_26":"Bend both knees equally — lower your hips with control.","correction_27":"Drop your weight through both legs — equal depth in each knee.","correction_28":"Both knees want more bend — this is a squat, commit to it.","correction_29":"Straighten both legs — press both heels down and lengthen.","correction_30":"Both legs can extend more — fire the quads and feel the length.","correction_31":"Open your left hip — let it rotate outward gently.","correction_32":"Your left hip is too closed — rotate it open and give it space.","correction_33":"Left hip needs more external rotation — let it spiral outward.","correction_34":"In this pose your left hip opens to the side — allow that rotation.","correction_35":"Left hip is restricted — breathe in and let it release outward.","correction_36":"Square your left hip forward — draw it back toward center.","correction_37":"Left hip is swinging open — bring it forward and stack it.","correction_38":"Pull your left hip in — imagine headlights facing forward.","correction_39":"Left hip rotates inward here — close it down slightly.","correction_40":"Open your right hip — rotate it out and let it breathe.","correction_41":"Right hip needs to open outward — allow the rotation.","correction_42":"Let your right hip externally rotate — imagine it turning like a door hinge.","correction_43":"Square your right hip — draw it forward to match the left.","correction_44":"Right hip is drifting open — bring it forward and level your pelvis.","correction_45":"Rotate your right hip inward just a touch — settle it.","correction_46":"Level your hips — both sides even, pelvis neutral.","correction_47":"Your hips are uneven — work to square them and find balance.","correction_48":"Hips want to melt down and back — let gravity pull them toward the heels.","correction_49":"Lift your left arm — sweep it up and reach through the fingertips.","correction_50":"Left arm is dropping — raise it to shoulder height and hold.","correction_51":"Float your left arm upward — there's more space to extend into.","correction_52":"Left arm needs to be parallel to the floor — extend it out level.","correction_53":"Arms in Warrior Two are wings — your left arm is falling, spread it wide.","correction_54":"Reach your left arm overhead — full extension through the wrist.","correction_55":"Lower your left arm — let it relax toward your side.","correction_56":"Your left arm is too high — ease it down and soften the shoulder.","correction_57":"Release the left shoulder down — unshrug it and let it drop.","correction_58":"Left shoulder is creeping up — draw the blade down your back.","correction_59":"Raise your right arm — extend it fully and reach out.","correction_60":"Right arm needs height — sweep it up and hold it there.","correction_61":"Your right arm is dropping below the line — extend it out level.","correction_62":"Sweep your right arm overhead — full reach from shoulder to fingertip.","correction_63":"Lower your right arm — let the shoulder soften and drop.","correction_64":"Right arm is too elevated — ease it down without collapsing the shape.","correction_65":"Release your right shoulder — slide the blade down and let the arm rest.","correction_66":"Sweep both arms up — reach them high overhead.","correction_67":"Both arms need to lift — raise them together, even and strong.","correction_68":"Both arms spread to the sides — like wings at shoulder height.","correction_69":"Release both shoulders — let them melt away from your ears.","correction_70":"Draw both shoulder blades down your back — create length in the neck.","correction_71":"Extend your left arm — press out through the heel of your hand.","correction_72":"Your left elbow is bent — lengthen through the arm.","correction_73":"Left arm needs more extension — straighten it without locking the joint.","correction_74":"Reach through your left fingertips — let the arm lengthen fully.","correction_75":"Soften your left elbow — let a gentle bend live there.","correction_76":"Your left arm is locking out — add a micro-bend to protect the joint.","correction_77":"Release the hyper-extension in your left arm — ease the elbow slightly.","correction_78":"Straighten your right arm — extend it fully from shoulder to wrist.","correction_79":"Your right elbow is folding — press through the hand and lengthen.","correction_80":"Right arm needs more extension — reach it a little further.","correction_81":"Soften the right elbow — a micro-bend keeps the joint safe.","correction_82":"Right arm is over-extending — ease the elbow, don't force it straight.","correction_83":"Let your right elbow have a natural soft bend — it doesn't need to lock.","correction_84":"Both arms need more extension — press out through both hands.","correction_85":"Soften both elbows — let a gentle bend live in each arm.","correction_86":"Press the floor away — in Downward Dog, the push comes from your hands.","correction_87":"Hips drive upward in Down Dog — lift them high and back.","correction_88":"Pedal your heels toward the mat — one at a time, working toward the floor.","correction_89":"Press the floor away and lift your chest — open the heart forward.","correction_90":"Keep your elbows drawing in toward the ribs — cobra arms hug the body.","correction_91":"Extend through the arms in Upward Dog — straighten them fully.","correction_92":"Standing leg is your trunk — keep that left knee firm and straight.","correction_93":"In Tree your arms are branches — sweep them up and let them grow.","correction_94":"Plank is a straight line from head to heel — engage your core.","correction_95":"Press the floor away in Plank — arms straight, no sag in the hips.","correction_96":"Melt your hips back toward your heels — surrender fully into it.","correction_97":"Arms reach long in Child's Pose — walk your fingertips forward.","correction_98":"In Seated Forward Fold, lead with your chest — hinge from the hips, not the back.","correction_99":"Legs are active in this fold — engage your quads and flex your feet.","correction_100":"Mountain Pose is active stillness — press both feet and stand tall.","correction_101":"In Mountain, arms rest alongside the body — soften them down.","correction_102":"In Triangle, back leg is straight and strong — press that heel into the mat.","correction_103":"Top arm reaches to the sky in Triangle — extend it fully from the hip.","correction_104":"Standing leg is straight and strong in Warrior Three — lock that knee.","correction_105":"Arms reach forward in Warrior Three — extend them long past your ears.","correction_106":"Bend both knees and raise your arms — find those two movements at once.","correction_107":"Ground through the legs while you extend the arms — rooted and reaching.","correction_108":"Open the front hip while squaring the back — find that opposing rotation.","correction_109":"Breathe into the restriction and let it soften on the exhale.","correction_110":"You're nearly there — one more breath and make that final adjustment.","correction_111":"Stay with it — the body takes a moment to arrive in these shapes.","correction_112":"Left knee tracks over the second toe — aim for that alignment.","correction_113":"Right knee tracks over the second toe — find that precision.","correction_114":"Ground through your left heel — the back leg is your foundation.","correction_115":"Press your right heel into the mat — feel the energy travel up the leg.","correction_116":"Neutral pelvis — neither tilting forward nor tucking under.","correction_117":"Tuck your tailbone — bring length to the lower back.","correction_118":"Lift through the back of your left knee — quadricep engaged.","correction_119":"Soften your right knee — not every pose wants a locked leg.","correction_120":"Your left shoulder is climbing — consciously release it away from your ear.","correction_121":"Right shoulder is tense — let it drop and create space in your neck.","correction_122":"Left arm is the compass — extend it fully and it guides the whole pose.","correction_123":"Right arm reaches back with intention — don't let it wilt.","correction_124":"Feel the length from your left hip to your left fingertips.","correction_125":"Right side of the body is long — reach from hip to fingertip.","correction_126":"Micro-bend your left elbow — it's for joint safety, not a big movement.","correction_127":"Micro-bend your right elbow — protective, not visible from the outside.","correction_128":"Both hips are level here — don't let either side hike up.","correction_129":"Hip crease deepens as you sink — allow that.","correction_130":"The whole left side is one line — knee, hip, shoulder, aligned.","correction_131":"Major adjustment needed — left knee is way off, bring it right over the ankle.","correction_132":"Big correction right knee — it needs much more bend, drop your hips significantly.","correction_133":"Left arm is completely down — you need to lift it fully to shoulder height.","correction_134":"Hips are far from the target — lower significantly and open wide.","correction_135":"Almost perfect on the left knee — just a touch more depth.","correction_136":"Right knee is spot on — minor refinement, track it slightly more forward.","correction_137":"Left shoulder just slightly low — barely lift it.","correction_138":"Right shoulder is a fraction high — ease it down imperceptibly.","correction_139":"Tiny adjustment — left hip wants just a little more opening.","correction_140":"Micro-adjustment right hip — draw it forward the smallest amount."};
const EN_STEPS = {"step_mountain_0":"Stand with your feet together or hip-width apart.","step_mountain_1":"Press all four corners of your feet firmly into the ground.","step_mountain_2":"Engage your thighs and lift your kneecaps slightly.","step_mountain_3":"Lengthen your tailbone down and lift your chest.","step_mountain_4":"Relax your shoulders away from your ears, arms at your sides.","step_mountain_5":"Breathe steadily and hold for five to ten breaths.","step_forward_fold_0":"Stand with feet hip-width apart.","step_forward_fold_1":"Inhale and lengthen your spine.","step_forward_fold_2":"Exhale and hinge forward from your hips.","step_forward_fold_3":"Let your hands hang toward the floor or hold your elbows.","step_forward_fold_4":"Bend your knees generously if your hamstrings are tight.","step_forward_fold_5":"Relax your head and neck completely.","step_forward_fold_6":"Hold for five breaths, deepening with each exhale.","step_halfway_lift_0":"From Forward Fold, place fingertips on the floor or shins.","step_halfway_lift_1":"Inhale and lift your torso halfway up.","step_halfway_lift_2":"Create a flat back, parallel to the ground.","step_halfway_lift_3":"Draw your shoulders back and down.","step_halfway_lift_4":"Gaze slightly forward to keep the neck long.","step_halfway_lift_5":"Exhale back into your forward fold.","step_warrior_i_0":"Step your left foot back into a lunge, about three to four feet.","step_warrior_i_1":"Turn your back foot out to forty-five degrees.","step_warrior_i_2":"Bend your front knee to ninety degrees over the ankle.","step_warrior_i_3":"Square your hips toward the front of the mat.","step_warrior_i_4":"Raise both arms overhead, palms facing each other.","step_warrior_i_5":"Lift through the chest and keep the back leg strong.","step_warrior_i_6":"Hold for five breaths.","step_warrior_ii_0":"Step your feet wide apart, about four feet.","step_warrior_ii_1":"Turn your front foot forward and back foot slightly in.","step_warrior_ii_2":"Bend your front knee to ninety degrees, stacking it over the ankle.","step_warrior_ii_3":"Extend your arms parallel to the floor, one forward, one back.","step_warrior_ii_4":"Gaze over your front fingertips.","step_warrior_ii_5":"Keep your torso upright and your back leg straight and strong.","step_warrior_ii_6":"Hold for five to seven breaths.","step_warrior_iii_0":"Start in Mountain Pose or Warrior One.","step_warrior_iii_1":"Shift your weight onto your front foot.","step_warrior_iii_2":"Hinge forward at the hips, lifting your back leg as your torso descends.","step_warrior_iii_3":"Aim for your body to form a T-shape, parallel to the floor.","step_warrior_iii_4":"Extend your arms forward alongside your ears or hold them at your sides.","step_warrior_iii_5":"Engage your core and the standing leg fully.","step_warrior_iii_6":"Hold for three to five breaths.","step_chair_0":"Stand in Mountain Pose.","step_chair_1":"Raise your arms overhead.","step_chair_2":"Bend your knees as if sitting into an imaginary chair.","step_chair_3":"Keep your knees tracking over your toes, not caving inward.","step_chair_4":"Hold your chest up and lengthen your tailbone down.","step_chair_5":"Hold for five to eight breaths.","step_tree_0":"Stand in Mountain Pose.","step_tree_1":"Shift your weight onto your left foot.","step_tree_2":"Bend your right knee and place your right foot on the inner left thigh or calf.","step_tree_3":"Bring your hands to your heart in prayer or raise them overhead.","step_tree_4":"Fix your gaze on a steady point to help your balance.","step_tree_5":"Hold for five to eight breaths then switch sides.","step_triangle_0":"Stand with your feet wide apart, about three to four feet.","step_triangle_1":"Turn your right foot out ninety degrees and left foot slightly in.","step_triangle_2":"Extend your arms out to the sides at shoulder height.","step_triangle_3":"Hinge at your right hip and lower your right hand toward your right shin or the floor.","step_triangle_4":"Raise your left arm straight up, stacking it over your right shoulder.","step_triangle_5":"Keep both legs straight and strong.","step_triangle_6":"Hold for five breaths then switch sides.","step_downward_dog_0":"Start on your hands and knees, wrists under shoulders and knees under hips.","step_downward_dog_1":"Tuck your toes and lift your hips up and back.","step_downward_dog_2":"Straighten your legs as much as feels comfortable.","step_downward_dog_3":"Press your hands firmly into the mat and rotate your upper arms outward.","step_downward_dog_4":"Relax your head between your arms.","step_downward_dog_5":"Hold for five to ten breaths.","step_plank_0":"Start in Downward Dog or a push-up position.","step_plank_1":"Align your wrists directly under your shoulders.","step_plank_2":"Form a straight line from your head to your heels.","step_plank_3":"Engage your core and avoid letting your hips sag or rise.","step_plank_4":"Press through the balls of your feet.","step_plank_5":"Hold for five to ten breaths.","step_low_lunge_0":"From Downward Dog, step your right foot between your hands.","step_low_lunge_1":"Lower your left knee to the mat.","step_low_lunge_2":"Align your right knee over your right ankle.","step_low_lunge_3":"Lift your torso upright and raise your arms overhead.","step_low_lunge_4":"Sink your hips toward the floor to deepen the stretch.","step_low_lunge_5":"Hold for five breaths then switch sides.","step_high_lunge_0":"From Downward Dog, step your right foot between your hands.","step_high_lunge_1":"Keep your back leg straight and lifted off the floor.","step_high_lunge_2":"Align your right knee over your right ankle.","step_high_lunge_3":"Lift your torso upright and raise your arms overhead.","step_high_lunge_4":"Press through the back heel to keep the leg active.","step_high_lunge_5":"Hold for five breaths then switch sides.","step_cobra_0":"Lie face down with your legs straight behind you.","step_cobra_1":"Place your palms on the floor under your shoulders.","step_cobra_2":"Hug your elbows in toward your ribs.","step_cobra_3":"Inhale and press into your hands to lift your chest off the floor.","step_cobra_4":"Keep your hips and thighs on the mat.","step_cobra_5":"Hold for three to five breaths.","step_upward_dog_0":"Lie face down with your legs straight behind you.","step_upward_dog_1":"Place your palms under your shoulders.","step_upward_dog_2":"Inhale and straighten your arms to lift your torso and thighs off the mat.","step_upward_dog_3":"Roll over the toes so the tops of your feet press into the mat.","step_upward_dog_4":"Lift your chest and draw your shoulders back and down.","step_upward_dog_5":"Hold for three to five breaths.","step_child_0":"Start in a kneeling position with your big toes together.","step_child_1":"Widen your knees to about hip-width or wider.","step_child_2":"Exhale and fold forward, extending your arms along the mat.","step_child_3":"Rest your forehead on the mat.","step_child_4":"Breathe deeply and hold for ten to twenty breaths.","step_seated_forward_0":"Sit on the mat with your legs extended straight in front of you.","step_seated_forward_1":"Flex your feet and engage your thighs.","step_seated_forward_2":"Inhale and lengthen your spine.","step_seated_forward_3":"Exhale and hinge from your hips to fold forward over your legs.","step_seated_forward_4":"Reach toward your feet, holding your ankles or feet if possible.","step_seated_forward_5":"Hold for seven to ten breaths, deepening with each exhale.","step_cat_cow_0":"Start on your hands and knees in a tabletop position.","step_cat_cow_1":"Align your wrists under your shoulders and knees under hips.","step_cat_cow_2":"Inhale for Cow — drop your belly, lift your chest and tailbone.","step_cat_cow_3":"Exhale for Cat — round your spine toward the ceiling, tuck your chin and tailbone.","step_cat_cow_4":"Continue flowing between the two shapes with your breath.","step_cat_cow_5":"Repeat for five to ten breath cycles."};

const ALL_EN = { ...EN_STRINGS, ...EN_CORRECTIONS, ...EN_STEPS };
const ALL_KEYS = Object.keys(ALL_EN);
const ALL_VALS = Object.values(ALL_EN);
const CHUNK = 40;

async function translateChunk(keys, values, langName) {
  const prompt = `Translate these ${values.length} English strings to ${langName}. Return ONLY a valid JSON array of translated strings in the exact same order. Preserve any {placeholder} tokens like {score}, {pose_name}, {names} exactly as-is. No explanations, no markdown, just the JSON array.\n\n${JSON.stringify(values)}`;
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const data = await resp.json();
  const raw = data.content[0].text.trim().replace(/^```json\n?|^```\n?|\n?```$/g, "").trim();
  const arr = JSON.parse(raw);
  const out = {};
  keys.forEach((k, i) => { out[k] = arr[i] || values[i]; });
  return out;
}

function downloadJSON(lang, langName, strings) {
  const payload = { lang, lang_name: langName, total: Object.keys(strings).length, strings };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `translations_${lang}.json`;
  a.click();
}

export default function TranslationTool() {
  const [selected, setSelected] = useState(new Set(["es"]));
  const [status, setStatus] = useState({});   // lang -> { state: idle|running|done|error, progress, log }
  const [results, setResults] = useState({});
  const [busy, setBusy] = useState(false);
  const logRef = useRef({});

  const toggle = (code) => {
    if (busy) return;
    setSelected(prev => {
      const n = new Set(prev);
      n.has(code) ? n.delete(code) : n.add(code);
      return n;
    });
  };

  const addLog = (lang, msg) => {
    setStatus(prev => ({
      ...prev,
      [lang]: {
        ...prev[lang],
        log: [...(prev[lang]?.log || []), msg],
      }
    }));
  };

  const startTranslation = async () => {
    if (busy || !selected.size) return;
    setBusy(true);

    for (const lang of selected) {
      const langName = LANGS.find(l => l.code === lang).name;
      setStatus(prev => ({ ...prev, [lang]: { state: "running", progress: 0, log: [`Starting ${langName}...`] } }));

      const translated = {};
      let failed = 0;

      for (let i = 0; i < ALL_KEYS.length; i += CHUNK) {
        const chunkKeys = ALL_KEYS.slice(i, i + CHUNK);
        const chunkVals = ALL_VALS.slice(i, i + CHUNK);
        const chunkNum = Math.ceil(i / CHUNK) + 1;
        const totalChunks = Math.ceil(ALL_KEYS.length / CHUNK);

        try {
          const res = await translateChunk(chunkKeys, chunkVals, langName);
          Object.assign(translated, res);
          addLog(lang, `Chunk ${chunkNum}/${totalChunks} — ${chunkVals.length} strings done`);
        } catch (e) {
          failed += chunkVals.length;
          chunkKeys.forEach((k, j) => { translated[k] = chunkVals[j]; });
          addLog(lang, `Chunk ${chunkNum}/${totalChunks} — error: ${e.message} (kept English)`);
        }

        const progress = Math.round(((i + CHUNK) / ALL_KEYS.length) * 100);
        setStatus(prev => ({ ...prev, [lang]: { ...prev[lang], progress: Math.min(progress, 100) } }));
        await new Promise(r => setTimeout(r, 200));
      }

      const total = Object.keys(translated).length;
      setResults(prev => ({ ...prev, [lang]: { langName, strings: translated } }));
      setStatus(prev => ({
        ...prev,
        [lang]: {
          ...prev[lang],
          state: failed > 0 ? "done_partial" : "done",
          progress: 100,
          log: [...(prev[lang]?.log || []), `Done — ${total} strings translated${failed ? `, ${failed} kept English` : ""}`],
        }
      }));
    }

    setBusy(false);
  };

  const s = {
    wrap: { padding: "1rem 0" },
    header: { fontSize: 18, fontWeight: 500, margin: "0 0 4px" },
    sub: { fontSize: 13, color: "var(--color-text-secondary)", margin: "0 0 1.25rem" },
    grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8, marginBottom: 16 },
    langBtn: (code) => ({
      padding: "8px 10px", fontSize: 13, textAlign: "left", cursor: busy ? "default" : "pointer",
      background: selected.has(code) ? "var(--color-background-info)" : undefined,
      color: selected.has(code) ? "var(--color-text-info)" : undefined,
      border: selected.has(code) ? "0.5px solid var(--color-border-info)" : undefined,
      opacity: busy ? 0.7 : 1,
    }),
    startBtn: { padding: "8px 20px", fontSize: 14, opacity: busy || !selected.size ? 0.5 : 1, cursor: busy ? "default" : "pointer" },
    progressWrap: { marginTop: 16 },
    track: { height: 4, background: "var(--color-border-tertiary)", borderRadius: 2, overflow: "hidden", marginBottom: 4 },
    fill: (pct) => ({ height: "100%", width: pct + "%", background: "var(--color-text-info)", transition: "width 0.3s" }),
    langStatus: { marginTop: 12, padding: "10px 12px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", fontSize: 12 },
    logLine: { color: "var(--color-text-secondary)", lineHeight: 1.6 },
    doneTag: { color: "var(--color-text-success)", fontWeight: 500 },
    dlRow: { marginTop: 8, display: "flex", gap: 8, alignItems: "center" },
    dlBtn: { padding: "5px 14px", fontSize: 12 },
  };

  const activeLangs = LANGS.filter(l => selected.has(l.code) || status[l.code]);

  return (
    <div style={s.wrap}>
      <p style={s.header}>Yoga correction — translation generator</p>
      <p style={s.sub}>Translates all 351 spoken strings using Claude API. Select languages, click Translate. ~60–90s per language.</p>

      <div style={s.grid}>
        {LANGS.map(l => (
          <button key={l.code} style={s.langBtn(l.code)} onClick={() => toggle(l.code)}>
            {l.name}{results[l.code] ? " ✓" : ""}
          </button>
        ))}
      </div>

      <button style={s.startBtn} onClick={startTranslation} disabled={busy || !selected.size}>
        {busy ? "Translating..." : `Translate ${selected.size} language${selected.size !== 1 ? "s" : ""}`}
      </button>

      {activeLangs.filter(l => status[l.code]).map(l => {
        const st = status[l.code];
        return (
          <div key={l.code} style={s.langStatus}>
            <div style={{ fontWeight: 500, marginBottom: 4, fontSize: 13 }}>
              {l.name}
              {st.state === "done" && <span style={{ ...s.doneTag, marginLeft: 8 }}>✓ complete</span>}
              {st.state === "done_partial" && <span style={{ color: "var(--color-text-warning)", marginLeft: 8 }}>✓ partial</span>}
              {st.state === "running" && <span style={{ color: "var(--color-text-secondary)", marginLeft: 8 }}>{st.progress}%</span>}
            </div>

            <div style={s.track}><div style={s.fill(st.progress || 0)} /></div>

            <div style={{ maxHeight: 80, overflowY: "auto" }}>
              {(st.log || []).slice(-6).map((line, i) => (
                <div key={i} style={s.logLine}>{line}</div>
              ))}
            </div>

            {results[l.code] && (
              <div style={s.dlRow}>
                <button style={s.dlBtn} onClick={() => downloadJSON(l.code, l.name, results[l.code].strings)}>
                  Download translations_{l.code}.json
                </button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
