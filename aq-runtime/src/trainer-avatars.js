// Public server-validated roster: verified same-character front/back game sprite pairs.
// Keep asset names stable; old six class IDs are deliberately retired.
export const TRAINER_AVATARS=Object.freeze([
  {id:'dp-lucas',name:'コウキ（DP）',generation:4},
  {id:'dp-dawn',name:'ヒカリ（DP）',generation:4},
  {id:'dp-barry',name:'ジュン（DP）',generation:4},
  {id:'pt-lucas',name:'コウキ（Pt）',generation:4},
  {id:'pt-dawn',name:'ヒカリ（Pt）',generation:4},
  {id:'pt-barry',name:'ジュン（Pt）',generation:4},
  {id:'pt-cheryl',name:'モミ（Pt）',generation:4},
  {id:'pt-riley',name:'ゲン（Pt）',generation:4},
  {id:'pt-buck',name:'バク（Pt）',generation:4},
  {id:'pt-mira',name:'ミル（Pt）',generation:4},
  {id:'pt-marley',name:'マイ（Pt）',generation:4},
  {id:'hgss-ethan',name:'ヒビキ（HGSS）',generation:4},
  {id:'hgss-lyra',name:'コトネ（HGSS）',generation:4},
  {id:'hgss-silver',name:'シルバー（HGSS）',generation:4},
  {id:'hgss-lance',name:'ワタル（HGSS）',generation:4},
  {id:'bw-hilbert',name:'トウヤ（BW）',generation:5},
  {id:'bw-hilda',name:'トウコ（BW）',generation:5},
  {id:'bw-cheren',name:'チェレン（BW）',generation:5},
  {id:'b2w2-nate',name:'キョウヘイ（B2W2）',generation:5},
  {id:'b2w2-rosa',name:'メイ（B2W2）',generation:5},
  {id:'b2w2-hugh',name:'ヒュウ（B2W2）',generation:5},
  {id:'b2w2-bianca',name:'ベル（B2W2）',generation:5},
  {id:'b2w2-cheren',name:'チェレン（B2W2）',generation:5},
  {id:'b2w2-chili',name:'ポッド（B2W2）',generation:5},
  {id:'b2w2-cilan',name:'デント（B2W2）',generation:5},
  {id:'b2w2-cress',name:'コーン（B2W2）',generation:5},
].map(row=>Object.freeze({...row,src:'/trainers/'+row.id+'-front.png',backSrc:'/trainers/'+row.id+'-back.png'})));
const avatarIds=new Set(TRAINER_AVATARS.map(row=>row.id));
export function defaultTrainerAvatar(seat=0) {return seat===1?'pt-dawn':'pt-lucas';}
export function isTrainerAvatar(value) {return typeof value==='string'&&avatarIds.has(value);}
export function trainerAvatarId(value,seat=0) {return isTrainerAvatar(value)?value:defaultTrainerAvatar(seat);}
export function trainerAvatarImage(value,seat=0) {return '/trainers/'+trainerAvatarId(value,seat)+'-front.png';}
export function trainerAvatarBackImage(value,seat=0) {return '/trainers/'+trainerAvatarId(value,seat)+'-back.png';}
