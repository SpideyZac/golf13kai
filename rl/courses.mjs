// Which holes the agent trains on and is judged on.
//
// TRAINING plays random REMIX courses (seed >= TRAIN_SEED_MIN): the classic 18
// rows re-dealt with freshly rolled land, so the agent never memorises a
// layout. EVALUATION uses courses it has never trained on: the CLASSIC course
// (seed 1113, not remix) and remix seeds below TRAIN_SEED_MIN.
import { CLASSIC_SEED } from './obs.mjs';

export const TRAIN_SEED_MIN = 1000, TRAIN_SEED_MAX = 2e6;

// rand: () => [0,1). classicProb lets a run also practise the classic course.
export function trainEpisode(rand, classicProb = 0)
{
    const classic = rand() < classicProb;
    let seed = CLASSIC_SEED;
    if (!classic)
        do seed = TRAIN_SEED_MIN + Math.floor(rand()*(TRAIN_SEED_MAX - TRAIN_SEED_MIN));
        while (seed == CLASSIC_SEED);
    return {seed, remix: !classic, hole: Math.floor(rand()*18), rngSeed: 1 + Math.floor(rand()*2**31)};
}

// Full 18-hole rounds. rounds = number of wind draws (classic) or courses (remix).
export function evalSet(name, rounds)
{
    const eps = [];
    for (let r = 0; r < rounds; ++r)
        for (let h = 0; h < 18; ++h)
            eps.push(name == 'classic'
                ? {seed: CLASSIC_SEED, remix: false, hole: h, rngSeed: 1000*(r+1) + h, round: r}
                : {seed: r + 1, remix: true, hole: h, rngSeed: 1000*(r+1) + h, round: r});
    return eps;
}
