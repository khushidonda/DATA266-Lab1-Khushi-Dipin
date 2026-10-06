# Kaggle evidence (Khushi, Task 3)

`kaggle_leaderboard_rank29_score_-49.5121.png` is the unaltered screenshot of the class-competition leaderboard row. What it shows (and nothing more is read into it):
- Rank 29, team name `PairProgramming_Team_1`, score `-49.5121`.
- Banner: "New personal best! Your latest submission scored -49.5121, improving on your previous best of -49.8366."
- The two unlabeled columns in the crop (values `3` and `5d`) are not interpreted here.

Cross-check, computed (not from the screenshot): the submitted file `../submission.csv` (`ID,FID,MiFID` / `1,98.61422779159997,0.40998050570487976`) gives -(FID + MiFID)/2 = -49.51210415, which rounds to the displayed -49.5121. The submitted file comes from checkpoint `epoch_28.pt` (completed epoch 29, SHA-256 `d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c`); see `../../../../reproducibility/manifests/khushi/task3_checkpoint_result_map.md`.

The rank is the leaderboard position at the time of the screenshot and can change as other teams submit.
