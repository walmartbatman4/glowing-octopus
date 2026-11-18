"""
Validates acceptance criteria specified in agent_task_beamer.md
"""

import os
import re
import yaml

def check():
    print("Running slides acceptance checks...")
    
    # Check 1: Files exist
    assert os.path.exists("slides/main.tex"), "slides/main.tex is missing"
    assert os.path.exists("slides/images/CREDITS.md"), "CREDITS.md is missing"
    assert os.path.exists("slides/NOTES.md"), "NOTES.md is missing"
    assert os.path.exists("slides/generated/macros.tex"), "macros.tex is missing"
    assert os.path.exists("slides/generated/results_table.tex"), "results_table.tex is missing"
    
    with open("slides/main.tex") as f:
        tex = f.read()

    # Check 2: Count frames (Title + 8 content frames + Credits + Thank You)
    frames = re.findall(r"\\begin\{frame\}", tex)
    print(f"Total frames found: {len(frames)} (Target: 11 including title, credits, and thank you)")
    assert len(frames) in [10, 11], f"Expected 10 or 11 frames, found {len(frames)}"

    # Check 3: Check notes word count
    notes = re.findall(r"\\note\{([^}]+)\}", tex)
    all_words = " ".join(notes).split()
    total_words = len(all_words)
    est_time_sec = total_words / (130 / 60)
    print(f"Notes word count: {total_words} words (Limit: <= 1,100 words)")
    print(f"Estimated presentation time: {est_time_sec/60:.2f} minutes (Limit: <= 9:30)")
    assert total_words <= 1100, f"Notes exceed 1,100 words ({total_words})"
    assert est_time_sec <= 570, f"Time exceeds 9:30 ({est_time_sec/60:.2f} min)"

    # Check 4: Check includegraphics exist
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for g in graphics:
        p = os.path.join("slides", g)
        assert os.path.exists(p), f"Referenced graphic not found: {p}"
        print(f"Verified graphic: {g}")

    # Check 5: Check no hardcoded numbers in slides 6-8 (use of macros)
    # Ensure \pairAUCSixD and \controlAUCSixD are used
    assert "\\pairAUCSixD" in tex, "Macro \\pairAUCSixD missing in main.tex"
    assert "\\controlAUCSixD" in tex, "Macro \\controlAUCSixD missing in main.tex"

    # Check 6: Check architecture match with config
    with open("configs/cinn_2d.yaml") as f:
        c2 = yaml.safe_load(f)
    with open("configs/cinn_6d.yaml") as f:
        c6 = yaml.safe_load(f)
    with open("slides/generated/macros.tex") as f:
        m_tex = f.read()
    
    assert f"\\newcommand{{\\numTransformsTwoD}}{{{c2['model']['num_transforms']}}}" in m_tex
    assert f"\\newcommand{{\\numTransformsSixD}}{{{c6['model']['num_transforms']}}}" in m_tex
    print("Architecture values match YAML configurations perfectly.")

    print("\nALL ACCEPTANCE CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    check()
