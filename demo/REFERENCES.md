# Visual references for Fly Stories

Reviewed in the browser on 2026-09-15, including playback of the X clips and the Hugging Face demo's Walk and Fly controls.

| Reference | What is visible | Use in our demo |
| --- | --- | --- |
| [Fly playing Mario](https://x.com/barrelshifter/status/2097004115826200898) | Gameplay beside an anatomical neural view, input image and activity indicators. | Keep the generated output and its corresponding recorded activity visible together. |
| [Navigation and homing](https://x.com/BrainsAndTennis/status/2099561733056794645) | A fly moves through a gridded world near bananas, with colored paths and neural/state diagrams. | Give the fly visible movement through space, with a bounded route and clear turns. |
| [Fly at a keyboard](https://x.com/insomnia_vip/status/2099583641462456830) | A code editor fills while an animated fly stands at a keyboard and a separate brain view changes. | Let the fly perform alongside the emerging story; retain a readable output panel and a distinct activity view. |
| [Robot in a maze](https://x.com/ZentrixHQ/status/2099704924032393694) | A physical wheeled robot travels between cardboard walls, with a neural visualization inset. | Treat the action and activity as a coordinated presentation. This clip's actor is a robot. |
| [Xenova Neural Canvas](https://huggingface.co/spaces/Xenova/fruit-fly-simulation) | An anatomical point cloud beside a detailed fly with articulated walking, turning and escape flight. | Reuse the already bundled, attributed NeuroMechFly rig and motion conventions; make movement discoverable during story playback. |

These are observations about the presentations. Social captions do not establish biological intelligence or explain the complete causal machinery behind a clip.

Our rank128 checkpoint produces language and recurrent states. The displayed states are recorded model activations. Story movement is illustrative choreography; it is not a trained motor policy or a measurement of a living fly. It must never alter, invent or rescale individual recorded neural frames to suit the choreography.

The body rig and the existing animation implementation are already attributed in `public/assets/Body-MIT.txt`, `public/assets/Xenova-LICENSE.txt`, and the provenance in `public/assets/fly-rig.json`. No video has been downloaded or bundled.
