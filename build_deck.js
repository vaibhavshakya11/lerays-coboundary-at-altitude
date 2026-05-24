// build_deck_v2.js
const pptxgen = require("pptxgenjs");
const fs   = require("fs");
const path = require("path");
const React           = require("react");
const ReactDOMServer  = require("react-dom/server");
const sharp           = require("sharp");
const FA = require("react-icons/fa");

const COLOR = {
  navy:       "1E3A5F",
  navyDark:   "152B47",
  navyLight:  "2C5078",
  gold:       "B5894C",
  goldLite:   "D9B47E",
  brick:      "A04040",
  brickLite:  "C97070",
  forest:     "3A6B3A",
  forestLite: "6B9B6B",
  cream:      "F4F1EA",
  paper:      "FCFAF6",
  ink:        "1A1F2E",
  inkSoft:    "47536A",
  ruleLite:   "DCD4C2",
  shadow:     "B0A892",
};

const FONT_HEAD = "Georgia";
const FONT_BODY = "Calibri";

const GITHUB_URL = "github.com/vaibhavshakya11/lerays-coboundary-at-altitude";

// ============================================================
// Helpers
// ============================================================
async function iconPng(IconComp, color = "#1E3A5F", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComp, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

function loadPdfAsImage(pdfPath, outPng) {
  const { execSync } = require("child_process");
  const tmpBase = outPng.replace(/\.png$/, "");
  execSync(`pdftoppm -png -r 220 -singlefile "${pdfPath}" "${tmpBase}"`);
  return outPng;
}

function vrule(slide, x, y, h, color = COLOR.ruleLite, width = 0.75) {
  slide.addShape("line", { x, y, w: 0, h, line: { color, width } });
}

function hrule(slide, x, y, w, color = COLOR.ruleLite, width = 0.75) {
  slide.addShape("line", { x, y, w, h: 0, line: { color, width } });
}

function footer(slide, pageNum, total) {
  slide.addText("Leray's Coboundary at Altitude  ·  Vaibhav Shakya  ·  GRC 2026", {
    x: 0.4, y: 5.30, w: 7.0, h: 0.2,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.inkSoft, align: "left", margin: 0,
  });
  slide.addText(`${pageNum} / ${total}`, {
    x: 8.6, y: 5.30, w: 1.0, h: 0.2,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.inkSoft, align: "right", margin: 0,
  });
}

function slideTitle(slide, title, kicker, titleH = 0.50) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: 0.4, y: 0.30, w: 9.2, h: 0.22,
      fontFace: FONT_BODY, fontSize: 9, color: COLOR.gold, bold: true,
      charSpacing: 4, margin: 0,
    });
  }
  const titleY = kicker ? 0.52 : 0.40;
  slide.addText(title, {
    x: 0.4, y: titleY, w: 9.2, h: titleH,
    fontFace: FONT_HEAD, fontSize: 20, bold: true, color: COLOR.ink, margin: 0,
  });
  hrule(slide, 0.4, titleY + titleH + 0.05, 9.2, COLOR.ruleLite);
}

// ============================================================
const FIG = path.join(__dirname, "figures");
const FIG_PNG_DIR = path.join(__dirname, "deck_figs");
if (!fs.existsSync(FIG_PNG_DIR)) fs.mkdirSync(FIG_PNG_DIR);

const figs = [
  "fig01_frontend_matrix", "fig06_multifault", "fig07_altitude_bound",
  "fig08_common_mode", "fig11_mission_summary", "fig14_ldpc_separation",
  "fig15_composed", "fig16_runtime_memory",
];
for (const f of figs) {
  const pdf = path.join(FIG, `${f}.pdf`);
  const png = path.join(FIG_PNG_DIR, `${f}.png`);
  if (fs.existsSync(pdf) && !fs.existsSync(png)) loadPdfAsImage(pdf, png);
}

// ============================================================
async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Vaibhav Shakya";
  pres.title  = "Leray's Coboundary at Altitude";

  const TOTAL = 16;

  const iconCheck = await iconPng(FA.FaCheckCircle, "#3A6B3A");
  const iconCross = await iconPng(FA.FaTimesCircle, "#A04040");
  const iconWarn  = await iconPng(FA.FaExclamationTriangle, "#B5894C");
  const wCar      = await iconPng(FA.FaCar, "#FFFFFF");
  const wHeart    = await iconPng(FA.FaHeartbeat, "#FFFFFF");
  const wCog      = await iconPng(FA.FaCog, "#FFFFFF");
  const iconGit   = await iconPng(FA.FaGithub, "#D9B47E");

  // ===========================================================
  // SLIDE 1 — TITLE
  // ===========================================================
  let s = pres.addSlide();
  s.background = { color: COLOR.navyDark };
  // Topological motif
  s.addShape("ellipse", { x: 7.4, y: 0.6, w: 2.5, h: 2.5,
      fill: { color: COLOR.navyLight, transparency: 70 },
      line: { color: COLOR.gold, width: 0.5 } });
  s.addShape("ellipse", { x: 7.9, y: 1.5, w: 2.0, h: 2.0,
      fill: { color: COLOR.gold, transparency: 80 },
      line: { color: COLOR.gold, width: 0.5 } });
  s.addShape("ellipse", { x: 7.0, y: 1.8, w: 1.4, h: 1.4,
      fill: { color: COLOR.navy, transparency: 50 },
      line: { color: COLOR.gold, width: 0.5 } });
  // a few edges to suggest a sheaf-graph
  s.addShape("line", { x: 7.5, y: 1.0, w: 0.5, h: 1.2,
      line: { color: COLOR.gold, width: 0.6 } });
  s.addShape("line", { x: 8.5, y: 2.2, w: -1.0, h: 0.3,
      line: { color: COLOR.gold, width: 0.6 } });

  s.addText("Synthica × NSRI Global Research Challenge", {
    x: 0.6, y: 0.5, w: 6.6, h: 0.3,
    fontFace: FONT_BODY, fontSize: 10.5, color: COLOR.goldLite, bold: true, charSpacing: 3, margin: 0,
  });
  s.addText("Leray's", {
    x: 0.6, y: 1.1, w: 8.5, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 44, bold: true, italic: true, color: "FFFFFF", margin: 0,
  });
  s.addText("Coboundary at Altitude", {
    x: 0.6, y: 1.75, w: 8.5, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 44, bold: true, color: "FFFFFF", margin: 0,
  });
  hrule(s, 0.6, 2.62, 1.4, COLOR.gold, 1.8);
  s.addText("A cellular-sheaf compiler framework for software-defined\nradiation hardness on deep-space flight computers", {
    x: 0.6, y: 2.78, w: 7.5, h: 0.9,
    fontFace: FONT_BODY, fontSize: 14, color: "E0DCD0", margin: 0,
  });

  s.addText("Vaibhav Shakya", {
    x: 0.6, y: 4.3, w: 6.0, h: 0.3,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("Jayshree Periwal International School, Jaipur", {
    x: 0.6, y: 4.62, w: 6.5, h: 0.25,
    fontFace: FONT_BODY, fontSize: 10.5, color: "B6BFD0", margin: 0,
  });
  s.addImage({ data: iconGit, x: 0.6, y: 4.95, w: 0.18, h: 0.18 });
  s.addText(GITHUB_URL, {
    x: 0.85, y: 4.93, w: 6.0, h: 0.22,
    fontFace: "Consolas", fontSize: 10, color: COLOR.goldLite, margin: 0,
  });

  // ===========================================================
  // SLIDE 2 — THE PROBLEM (with TMR voter diagram)
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Why deep-space computers triplicate every calculation", "background", 0.50);

  // Left: explanation paragraphs
  s.addText("Every flight computer beyond low Earth orbit faces single-event upsets — heavy ions and protons that flip individual bits in memory or registers. The standard defence is triple modular redundancy (TMR): run every calculation three times and take a majority vote.", {
    x: 0.4, y: 1.30, w: 5.4, h: 1.3,
    fontFace: FONT_BODY, fontSize: 12, color: COLOR.ink, margin: 0,
  });

  s.addText("TMR has carried Voyager, Cassini, and Europa Clipper. It is also wasteful. Three replicas burn three times the compute and three times the power. Worse, when a single ion clips all three replicas in the same way — a common-mode strike — the majority vote is unanimous and wrong.", {
    x: 0.4, y: 2.70, w: 5.4, h: 1.5,
    fontFace: FONT_BODY, fontSize: 12, color: COLOR.ink, margin: 0,
  });

  s.addText("Heavy-ion data puts common-mode strikes at 1–10% of all upsets. Small fraction, catastrophic consequence.", {
    x: 0.4, y: 4.35, w: 5.4, h: 0.75,
    fontFace: FONT_BODY, fontSize: 11.5, italic: true, color: COLOR.brick, margin: 0, bold: true,
  });

  // Right: TMR voter diagram
  // Three replica boxes feeding into a voter
  const rx = 6.4;
  s.addText("How TMR works", {
    x: rx, y: 1.30, w: 3.5, h: 0.30,
    fontFace: FONT_HEAD, fontSize: 12, bold: true, color: COLOR.navy, margin: 0,
  });
  // Three computation boxes
  for (let i = 0; i < 3; i++) {
    const y = 1.75 + i * 0.65;
    const isHit = (i === 1);  // middle one shown as faulted
    s.addShape("roundRect", { x: rx, y, w: 1.3, h: 0.5,
        fill: { color: isHit ? COLOR.brickLite : "FFFFFF" },
        line: { color: isHit ? COLOR.brick : COLOR.ruleLite, width: 1 },
        rectRadius: 0.05 });
    s.addText(`replica ${i+1}`, {
      x: rx, y, w: 1.3, h: 0.5,
      fontFace: FONT_BODY, fontSize: 10, color: isHit ? "FFFFFF" : COLOR.ink,
      bold: true, align: "center", valign: "middle", margin: 0,
    });
    // arrow to voter
    s.addShape("line", {
      x: rx + 1.32, y: y + 0.25, w: 0.55, h: 1.20 - i * 0.45 - 0.05,
      line: { color: COLOR.inkSoft, width: 1.2, endArrowType: "triangle" },
    });
  }
  // Lightning bolt on the faulted one
  s.addText("⚡", {
    x: rx + 1.0, y: 2.30, w: 0.4, h: 0.4,
    fontFace: FONT_HEAD, fontSize: 18, color: COLOR.gold,
    align: "center", valign: "middle", margin: 0,
  });
  // Voter
  s.addShape("roundRect", { x: rx + 1.9, y: 2.30, w: 1.5, h: 0.55,
      fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 },
      rectRadius: 0.05 });
  s.addText("majority\nvote", {
    x: rx + 1.9, y: 2.30, w: 1.5, h: 0.55,
    fontFace: FONT_BODY, fontSize: 10, color: "FFFFFF", bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  // Output
  s.addShape("line", {
    x: rx + 3.42, y: 2.575, w: 0.30, h: 0,
    line: { color: COLOR.inkSoft, width: 1.2, endArrowType: "triangle" },
  });
  s.addText("output", {
    x: rx + 1.9, y: 2.95, w: 1.5, h: 0.25,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.inkSoft,
    align: "center", margin: 0,
  });

  // The common-mode failure
  s.addShape("rect", { x: rx - 0.05, y: 4.05, w: 3.55, h: 1.05,
      fill: { color: "FFFFFF" }, line: { color: COLOR.brick, width: 1 } });
  s.addShape("rect", { x: rx - 0.05, y: 4.05, w: 0.08, h: 1.05,
      fill: { color: COLOR.brick }, line: { color: COLOR.brick, width: 0 } });
  s.addText("Common-mode strike", {
    x: rx + 0.1, y: 4.12, w: 3.3, h: 0.25,
    fontFace: FONT_HEAD, fontSize: 11, bold: true, color: COLOR.brick, margin: 0,
  });
  s.addText("All three replicas flipped identically. Vote is unanimous — and wrong. Adding more replicas does not help.", {
    x: rx + 0.1, y: 4.38, w: 3.3, h: 0.7,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, margin: 0,
  });

  vrule(s, 6.15, 1.30, 3.8, COLOR.ruleLite);

  footer(s, 2, TOTAL);

  // ===========================================================
  // SLIDE 3 — THREE PRIOR FAMILIES
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Forty years of partial fixes", "the prior art");

  s.addText("Three families of alternatives have been proposed since the 1980s. Each addresses one weakness of TMR and leaves at least one other unsolved. None encodes program semantics directly; each new computational kernel requires fresh analysis.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.75,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const families = [
    { name: "Hamming / SECDED",
      what: "Bit-level parity codes\n(R. W. Hamming, 1950)",
      pro:  "Cheap: ~30% storage overhead",
      con:  "Blind to faults wider than 2 bits. Production data: ~26% of strikes exceed the correction distance.",
    },
    { name: "Algorithm-Based Fault Tolerance",
      what: "Algebraic kernel checksums\n(Huang & Abraham, 1984)",
      pro:  "Handles multi-bit and common-mode faults",
      con:  "Per-kernel analysis. Each new algorithm needs custom invariants designed by hand.",
    },
    { name: "SWIFT / instruction duplication",
      what: "Selective instruction replication\n(Reis et al., 2005)",
      pro:  "Matches TMR coverage at lower overhead",
      con:  "Still vulnerable to correlated strikes — inherits the failure mode TMR cannot fix.",
    },
  ];
  const cardW = 3.05, cardH = 3.0, gap = 0.15;
  let cardX = 0.4;
  for (const f of families) {
    s.addShape("rect", { x: cardX, y: 2.05, w: cardW, h: cardH,
        fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addText(f.name, {
      x: cardX + 0.2, y: 2.15, w: cardW - 0.4, h: 0.38,
      fontFace: FONT_HEAD, fontSize: 13, bold: true, color: COLOR.navy, margin: 0,
    });
    s.addText(f.what, {
      x: cardX + 0.2, y: 2.55, w: cardW - 0.4, h: 0.50,
      fontFace: FONT_BODY, fontSize: 10, italic: true, color: COLOR.inkSoft, margin: 0,
    });
    hrule(s, cardX + 0.2, 3.10, cardW - 0.4, COLOR.ruleLite);

    s.addImage({ data: iconCheck, x: cardX + 0.2, y: 3.25, w: 0.22, h: 0.22 });
    s.addText(f.pro, {
      x: cardX + 0.5, y: 3.22, w: cardW - 0.65, h: 0.45,
      fontFace: FONT_BODY, fontSize: 10.5, color: COLOR.ink, bold: true, margin: 0,
    });

    s.addImage({ data: iconCross, x: cardX + 0.2, y: 3.75, w: 0.22, h: 0.22 });
    s.addText(f.con, {
      x: cardX + 0.5, y: 3.72, w: cardW - 0.65, h: 1.15,
      fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, margin: 0,
    });

    cardX += cardW + gap;
  }

  footer(s, 3, TOTAL);

  // ===========================================================
  // SLIDE 4 — CORE IDEA
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "A cellular sheaf assigns algebra to the control-flow graph", "the construction", 0.50);

  // Left explanation
  s.addText("Cellular sheaves come from algebraic topology (Curry 2014, Hansen & Ghrist 2019). For our purposes the definition is concrete: take the control-flow graph G of a program. Attach a vector space to each vertex (the program state at that point) and a vector space to each edge (the constraints that should hold across the transition). Attach a linear map to each vertex-edge incidence.", {
    x: 0.4, y: 1.20, w: 5.3, h: 2.0,
    fontFace: FONT_BODY, fontSize: 11.5, color: COLOR.ink, margin: 0,
  });

  // Coboundary box
  s.addShape("rect", { x: 0.4, y: 3.30, w: 5.3, h: 1.7,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addShape("rect", { x: 0.4, y: 3.30, w: 0.08, h: 1.7,
      fill: { color: COLOR.gold }, line: { color: COLOR.gold, width: 0 } });
  s.addText("The coboundary operator, Leray 1942", {
    x: 0.62, y: 3.38, w: 5.0, h: 0.28,
    fontFace: FONT_HEAD, fontSize: 11.5, bold: true, color: COLOR.navy, margin: 0,
  });
  s.addText("(δx)_e  =  ℱ_{u ⊴ e}(x_u)  −  ℱ_{v ⊴ e}(x_v)", {
    x: 0.62, y: 3.70, w: 5.0, h: 0.40,
    fontFace: "Consolas", fontSize: 13, color: COLOR.ink, italic: true, margin: 0,
  });
  s.addText("For each edge e = (u, v), the coboundary measures whether the state at vertex u maps to the same edge-stalk value as the state at vertex v. If a radiation event corrupts x somewhere, δx becomes non-zero in a structured way that identifies what went wrong.", {
    x: 0.62, y: 4.15, w: 5.0, h: 0.80,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.inkSoft, margin: 0,
  });

  // Right: sheaf graph with labeled stalks
  const vx = 6.5, vy = 1.40;
  const verts = [
    { x: vx + 0.0, y: vy + 0.7,  lbl: "v₁" },
    { x: vx + 1.5, y: vy + 0.1,  lbl: "v₂" },
    { x: vx + 2.7, y: vy + 0.9,  lbl: "v₃" },
    { x: vx + 1.9, y: vy + 2.2,  lbl: "v₄" },
    { x: vx + 0.3, y: vy + 2.0,  lbl: "v₅" },
  ];
  const edges = [[0,1],[1,2],[2,3],[3,4],[4,0],[1,3]];
  for (const [a,b] of edges) {
    s.addShape("line", {
      x: verts[a].x + 0.22, y: verts[a].y + 0.22,
      w: verts[b].x - verts[a].x, h: verts[b].y - verts[a].y,
      line: { color: COLOR.navyLight, width: 1.5 },
    });
  }
  for (const v of verts) {
    s.addShape("ellipse", { x: v.x, y: v.y, w: 0.44, h: 0.44,
        fill: { color: COLOR.navy }, line: { color: COLOR.gold, width: 1.2 } });
  }
  // Detail callout: show one vertex's stalk
  s.addShape("line", { x: vx + 0.44, y: vy + 0.92, w: 0.65, h: -0.3,
      line: { color: COLOR.inkSoft, width: 0.6, dashType: "dash" } });
  s.addShape("rect", { x: vx + 1.10, y: vy + 0.40, w: 1.55, h: 0.55,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addText("stalk ℱ(v₁) = ℝ^kv", {
    x: vx + 1.13, y: vy + 0.45, w: 1.5, h: 0.22,
    fontFace: "Consolas", fontSize: 9, color: COLOR.ink, margin: 0,
  });
  s.addText("k_v scalars of state", {
    x: vx + 1.13, y: vy + 0.68, w: 1.5, h: 0.22,
    fontFace: FONT_BODY, fontSize: 8.5, italic: true, color: COLOR.inkSoft, margin: 0,
  });

  s.addText("Each black node holds a k_v-dimensional state vector.\nEach edge holds a k_e-dim constraint vector. Restriction maps glue them.", {
    x: 5.9, y: 4.10, w: 4.0, h: 0.80,
    fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: COLOR.inkSoft,
    align: "center", margin: 0,
  });

  footer(s, 4, TOTAL);

  // ===========================================================
  // SLIDE 5 — MASTER PIPELINE
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "How the framework runs, from source code to fault recovery", "the pipeline", 0.50);

  s.addText("A compile-time pass synthesises the parity-check matrix H once, from the program's source. At runtime the protected program computes its syndrome (Hx − b) every iteration; if the syndrome exceeds a numerical threshold, an orthogonal matching pursuit decoder identifies which state slot was corrupted and recovers the clean value.", {
    x: 0.4, y: 1.20, w: 9.2, h: 0.95,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  s.addText("COMPILE TIME", {
    x: 0.4, y: 2.30, w: 9.2, h: 0.22,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.gold, bold: true, charSpacing: 4, margin: 0,
  });
  const ct = [
    { t: "source\nprogram",   c: "F4EDE0", tx: COLOR.ink },
    { t: "CFG  G",            c: "E5DFD0", tx: COLOR.ink },
    { t: "frontend\n(ℱ_{v⊴e})", c: "E5DFD0", tx: COLOR.ink },
    { t: "sheaf ℱ\non G",     c: COLOR.navy, tx: "FFFFFF" },
    { t: "parity\ncheck H",   c: COLOR.navy, tx: "FFFFFF" },
  ];
  let cx = 0.4;
  const boxW = 1.55, boxH = 0.75, gapW = 0.30;
  for (let i = 0; i < ct.length; i++) {
    const b = ct[i];
    s.addShape("roundRect", { x: cx, y: 2.55, w: boxW, h: boxH,
        fill: { color: b.c }, line: { color: COLOR.ruleLite, width: 0.5 }, rectRadius: 0.06 });
    s.addText(b.t, { x: cx, y: 2.55, w: boxW, h: boxH,
        fontFace: FONT_BODY, fontSize: 10.5, color: b.tx, align: "center", valign: "middle", bold: true, margin: 0 });
    if (i < ct.length - 1) {
      s.addShape("line", {
        x: cx + boxW + 0.04, y: 2.55 + boxH / 2,
        w: gapW - 0.08, h: 0,
        line: { color: COLOR.inkSoft, width: 1.5, endArrowType: "triangle" },
      });
    }
    cx += boxW + gapW;
  }

  s.addShape("line", {
    x: 4.95, y: 3.35, w: 0, h: 0.50,
    line: { color: COLOR.gold, width: 1.5, endArrowType: "triangle", dashType: "dash" },
  });
  s.addText("the same H", {
    x: 5.05, y: 3.45, w: 1.5, h: 0.25,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.gold, margin: 0,
  });

  s.addText("RUNTIME", {
    x: 0.4, y: 3.90, w: 9.2, h: 0.22,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.gold, bold: true, charSpacing: 4, margin: 0,
  });
  const rt = [
    { t: "state x\nat runtime",    c: COLOR.goldLite, tx: COLOR.ink },
    { t: "syndrome\ns = Hx − b",   c: COLOR.goldLite, tx: COLOR.ink },
    { t: "OMP\nrecover",           c: COLOR.goldLite, tx: COLOR.ink },
    { t: "clean  x̂",               c: COLOR.forest,   tx: "FFFFFF" },
  ];
  let rx2 = 0.4;
  const rBoxW = 1.95;
  for (let i = 0; i < rt.length; i++) {
    const b = rt[i];
    s.addShape("roundRect", { x: rx2, y: 4.15, w: rBoxW, h: boxH,
        fill: { color: b.c }, line: { color: COLOR.ruleLite, width: 0.5 }, rectRadius: 0.06 });
    s.addText(b.t, { x: rx2, y: 4.15, w: rBoxW, h: boxH,
        fontFace: FONT_BODY, fontSize: 10.5, color: b.tx, align: "center", valign: "middle", bold: true, margin: 0 });
    if (i < rt.length - 1) {
      s.addShape("line", {
        x: rx2 + rBoxW + 0.04, y: 4.15 + boxH / 2,
        w: 0.32, h: 0,
        line: { color: COLOR.inkSoft, width: 1.5, endArrowType: "triangle" },
      });
    }
    rx2 += rBoxW + 0.4;
  }

  footer(s, 5, TOTAL);

  // ===========================================================
  // SLIDE 6 — SIX FRONTENDS
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Six frontends cover linear, polynomial, neural, and statistical programs", "what the framework can protect", 0.50);

  s.addText("A frontend is a function that consumes a class of source program and emits the four-tuple (G, k_v, k_e, restriction maps). The same orthogonal matching pursuit decoder handles all six. Detection rates below are from the headline coverage matrix (n ≥ 130 per frontend, 95% Wilson confidence intervals).", {
    x: 0.4, y: 1.30, w: 9.2, h: 0.65,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const frontends = [
    { n: "Linear",            d: "Kalman filters, PID, FFT — state-space invariants",          r: "100%" },
    { n: "Polynomial",        d: "Quaternion norm, energy conservation — degree-d monomials",   r: "100%" },
    { n: "Piecewise linear",  d: "Saturation, ReLU — per-region linear maps",                   r: "100%" },
    { n: "Neural network",    d: "MLP inference — activation-mask + linear relations",          r: "100%" },
    { n: "Statistical",       d: "Black-box subsystems — PCA-derived ellipsoidal envelope",     r: "83.2%" },
    { n: "Nonlinear",         d: "Transcendental dynamics — Carleman linearisation",            r: "100%" },
  ];
  const fcW = 4.55, fcH = 0.85, fcGap = 0.13;
  for (let i = 0; i < frontends.length; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.4 + col * (fcW + fcGap);
    const y = 2.05 + row * (fcH + fcGap);
    const isStat = frontends[i].r !== "100%";
    s.addShape("rect", { x, y, w: fcW, h: fcH,
        fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addShape("rect", { x, y, w: 0.08, h: fcH,
        fill: { color: isStat ? COLOR.brick : COLOR.navy }, line: { color: "FFFFFF", width: 0 } });
    s.addText(frontends[i].n, {
      x: x + 0.20, y: y + 0.08, w: fcW - 1.3, h: 0.28,
      fontFace: FONT_HEAD, fontSize: 12.5, bold: true, color: COLOR.navy, margin: 0,
    });
    s.addText(frontends[i].d, {
      x: x + 0.20, y: y + 0.36, w: fcW - 1.3, h: 0.52,
      fontFace: FONT_BODY, fontSize: 10.5, color: COLOR.ink, margin: 0,
    });
    s.addText(frontends[i].r, {
      x: x + fcW - 1.20, y: y + 0.18, w: 1.10, h: 0.55,
      fontFace: FONT_HEAD, fontSize: 19, bold: true,
      color: isStat ? COLOR.brick : COLOR.forest,
      align: "right", valign: "middle", margin: 0,
    });
  }

  s.addText("The statistical frontend's 83.2% is structural. PCA-derived invariants have a numerical floor around 10⁻²; faults inside that noise envelope are statistically absorbed and produce no detectable syndrome.", {
    x: 0.4, y: 4.95, w: 9.2, h: 0.32,
    fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: COLOR.brick, margin: 0,
  });

  footer(s, 6, TOTAL);

  // ===========================================================
  // SLIDE 7 — THEOREM 1 (with degree visual)
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Theorem 1: code distance is set by the graph's degree sequence", "the altitude bound", 0.50);

  // Theorem box
  s.addShape("rect", { x: 0.4, y: 1.25, w: 9.2, h: 1.55,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addShape("rect", { x: 0.4, y: 1.25, w: 0.08, h: 1.55,
      fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 } });
  s.addText("Theorem 1 (Altitude bound)", {
    x: 0.65, y: 1.35, w: 8.8, h: 0.28,
    fontFace: FONT_HEAD, fontSize: 12, bold: true, color: COLOR.navy, margin: 0,
  });
  s.addText([
    { text: "Let G be a finite graph. Suppose there exists a ", options: { italic: true } },
    { text: "unique", options: { italic: true, bold: true } },
    { text: " vertex v₀ with deg(v₀) < k_v / k_e, and every other vertex satisfies deg(v) ≥ ⌈k_v / k_e⌉. Under continuous restriction maps,", options: { italic: true } },
  ], {
    x: 0.65, y: 1.65, w: 8.8, h: 0.55,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });
  s.addText("d(ℱ)  =  k_v       almost surely.", {
    x: 0.65, y: 2.20, w: 8.8, h: 0.35,
    fontFace: "Consolas", fontSize: 14, color: COLOR.navy, align: "center", bold: true, margin: 0,
  });
  s.addText("The minimum distance of the code equals the vertex-stalk dimension. Set by program graph structure alone — not the program's content.", {
    x: 0.65, y: 2.55, w: 8.8, h: 0.20,
    fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: COLOR.inkSoft, align: "center", margin: 0,
  });

  // Left lower: proof sketch
  s.addText("Two-part proof", {
    x: 0.4, y: 2.95, w: 5.5, h: 0.30,
    fontFace: FONT_HEAD, fontSize: 12, bold: true, color: COLOR.navy, margin: 0,
  });
  const proof = [
    "Upper bound: construct a weight-k_v codeword at v₀'s slack. Rank-nullity guarantees the kernel is at least 1-dimensional.",
    "Lower bound: case analysis on the support — single vertex at v ≠ v₀, single vertex at v₀, or multiple vertices. Each case ruled out under genericity.",
    "Exclusion set is a strict algebraic subvariety, hence measure-zero.",
  ];
  s.addText(proof.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < proof.length - 1 }
  })), {
    x: 0.4, y: 3.30, w: 5.3, h: 1.85,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, paraSpaceAfter: 4, margin: 0,
  });

  // Right lower: degree threshold diagram
  s.addText("Why the degree threshold matters", {
    x: 6.0, y: 2.95, w: 3.7, h: 0.30,
    fontFace: FONT_HEAD, fontSize: 11.5, bold: true, color: COLOR.navy, margin: 0,
  });

  // Small graph showing v0 with low degree (red) and others (green)
  const gx = 6.0, gy = 3.40;
  const dg = [
    { x: gx + 0.6,  y: gy + 0.4,  hi: false },
    { x: gx + 1.8,  y: gy + 0.2,  hi: false },
    { x: gx + 3.0,  y: gy + 0.5,  hi: false },
    { x: gx + 0.3,  y: gy + 1.4,  hi: true },   // v0 - the slack
    { x: gx + 2.6,  y: gy + 1.5,  hi: false },
  ];
  // Edges
  const dEdges = [[0,1],[1,2],[1,4],[2,4],[0,3]];
  for (const [a,b] of dEdges) {
    s.addShape("line", {
      x: dg[a].x + 0.14, y: dg[a].y + 0.14,
      w: dg[b].x - dg[a].x, h: dg[b].y - dg[a].y,
      line: { color: COLOR.navyLight, width: 1.0 },
    });
  }
  for (const v of dg) {
    s.addShape("ellipse", { x: v.x, y: v.y, w: 0.28, h: 0.28,
        fill: { color: v.hi ? COLOR.brick : COLOR.forest },
        line: { color: v.hi ? COLOR.brick : COLOR.forest, width: 0 } });
  }
  s.addText("v₀", {
    x: gx + 0.05, y: gy + 1.62, w: 0.5, h: 0.22,
    fontFace: "Consolas", fontSize: 9, bold: true, color: COLOR.brick, margin: 0,
  });
  s.addText("low deg = slack →\nhosts the shortest codeword", {
    x: gx + 0.65, y: gy + 1.40, w: 2.8, h: 0.40,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.inkSoft, margin: 0,
  });
  s.addText("high deg = constrained\n(no slack)", {
    x: gx + 1.4, y: gy + 0.55, w: 1.8, h: 0.30,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.forest, margin: 0,
  });

  footer(s, 7, TOTAL);

  // ===========================================================
  // SLIDE 8 — THEOREM 2 (with stalk-swap diagram)
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Theorem 2: real-valued restriction maps catch faults that binary parity cannot", "sheaf vs LDPC", 0.50);

  s.addShape("rect", { x: 0.4, y: 1.25, w: 9.2, h: 1.20,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addShape("rect", { x: 0.4, y: 1.25, w: 0.08, h: 1.20,
      fill: { color: COLOR.gold }, line: { color: COLOR.gold, width: 0 } });
  s.addText("Theorem 2 (Sheaf–LDPC separation)", {
    x: 0.65, y: 1.32, w: 8.8, h: 0.28,
    fontFace: FONT_HEAD, fontSize: 12, bold: true, color: COLOR.navy, margin: 0,
  });
  s.addText("Consider any binary LDPC code on the same Tanner graph as the sheaf. There exists a class of physically realistic faults — vertex-block scalar permutations — that the sheaf detects with probability one and that no binary LDPC code on the same support can detect.", {
    x: 0.65, y: 1.65, w: 8.8, h: 0.72,
    fontFace: FONT_BODY, fontSize: 11, italic: true, color: COLOR.ink, margin: 0,
  });

  // Left: stalk-swap diagram showing why
  s.addText("Why the swap is invisible to binary parity", {
    x: 0.4, y: 2.60, w: 5.5, h: 0.28,
    fontFace: FONT_HEAD, fontSize: 11.5, bold: true, color: COLOR.navy, margin: 0,
  });

  // Before/after stalk pair
  // Stalk before
  const sbx = 0.4, sby = 2.95;
  s.addText("before strike", {
    x: sbx, y: sby - 0.05, w: 2.0, h: 0.22,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.inkSoft, margin: 0,
  });
  // 4 stalk slots
  const slotsBefore = ["3.14", "2.71", "1.41", "0.58"];
  const slotsAfter  = ["2.71", "3.14", "1.41", "0.58"];  // first two swapped
  for (let i = 0; i < 4; i++) {
    const swapped = (i === 0 || i === 1);
    s.addShape("rect", { x: sbx + i * 0.55, y: sby + 0.20, w: 0.50, h: 0.45,
        fill: { color: swapped ? COLOR.brickLite : "FFFFFF" },
        line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addText(slotsBefore[i], {
      x: sbx + i * 0.55, y: sby + 0.20, w: 0.50, h: 0.45,
      fontFace: "Consolas", fontSize: 10, color: swapped ? "FFFFFF" : COLOR.ink,
      align: "center", valign: "middle", margin: 0,
    });
  }

  // After
  const sax = 0.4, say = 3.85;
  s.addText("after vertex-permutation strike (slot 1 ↔ slot 2)", {
    x: sax, y: say - 0.05, w: 4.5, h: 0.22,
    fontFace: FONT_BODY, fontSize: 9, italic: true, color: COLOR.inkSoft, margin: 0,
  });
  for (let i = 0; i < 4; i++) {
    const swapped = (i === 0 || i === 1);
    s.addShape("rect", { x: sax + i * 0.55, y: say + 0.20, w: 0.50, h: 0.45,
        fill: { color: swapped ? COLOR.brickLite : "FFFFFF" },
        line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addText(slotsAfter[i], {
      x: sax + i * 0.55, y: say + 0.20, w: 0.50, h: 0.45,
      fontFace: "Consolas", fontSize: 10, color: swapped ? "FFFFFF" : COLOR.ink,
      align: "center", valign: "middle", margin: 0,
    });
  }

  // Explanation
  s.addText("Sheaf check: each scalar is weighted by a distinct real-valued column. Swap → non-zero syndrome.", {
    x: 2.95, y: 3.05, w: 3.0, h: 0.55,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.forest, italic: true, margin: 0,
  });
  s.addText("Binary parity check: each scalar hashed to one bit (XOR-parity). If both scalars have the same parity bit, the swap is invisible.", {
    x: 2.95, y: 3.95, w: 3.0, h: 0.85,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.brick, italic: true, margin: 0,
  });

  // Right: chart + adversarial summary
  const ldpcImg = path.join(FIG_PNG_DIR, "fig14_ldpc_separation.png");
  if (fs.existsSync(ldpcImg)) {
    s.addImage({ path: ldpcImg, x: 6.0, y: 2.55, w: 3.7, h: 1.45 });
  }
  s.addShape("rect", { x: 6.0, y: 4.10, w: 3.7, h: 1.0,
      fill: { color: "FFFFFF" }, line: { color: COLOR.brick, width: 1 } });
  s.addShape("rect", { x: 6.0, y: 4.10, w: 0.08, h: 1.0,
      fill: { color: COLOR.brick }, line: { color: COLOR.brick, width: 0 } });
  s.addText("20 random binary LDPC codes constructed adversarially on the same support pattern. 40,000 vertex-permutation trials.", {
    x: 6.18, y: 4.16, w: 3.45, h: 0.50,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.ink, margin: 0,
  });
  s.addText("Sheaf: 100.00%   ·   Best random LDPC: 97.9%", {
    x: 6.18, y: 4.70, w: 3.45, h: 0.30,
    fontFace: "Consolas", fontSize: 10, bold: true, color: COLOR.navy, margin: 0,
  });

  footer(s, 8, TOTAL);

  // ===========================================================
  // SLIDE 9 — HEADLINE NUMBERS
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Empirical campaign: fifteen experiments, deterministic seeds", "results overview", 0.50);

  s.addText("Every experiment is driven by a fixed random seed and is reproducible from the released code with one command. Sample sizes were chosen per experiment to deliver 95% Wilson confidence intervals narrower than five percentage points on the primary outcome.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.70,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  function bigStat(slide, x, y, w, h, big, label, sub, color = COLOR.navy) {
    slide.addShape("rect", { x, y, w, h, fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    slide.addShape("rect", { x, y, w: 0.08, h, fill: { color }, line: { color, width: 0 } });
    slide.addText(big, {
      x: x + 0.18, y: y + 0.12, w: w - 0.3, h: h * 0.50,
      fontFace: FONT_HEAD, fontSize: 36, bold: true, color,
      align: "left", valign: "middle", margin: 0,
    });
    slide.addText(label, {
      x: x + 0.18, y: y + h * 0.58, w: w - 0.3, h: 0.28,
      fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, bold: true,
      align: "left", valign: "top", margin: 0,
    });
    if (sub) {
      slide.addText(sub, {
        x: x + 0.18, y: y + h * 0.80, w: w - 0.3, h: 0.22,
        fontFace: FONT_BODY, fontSize: 9, color: COLOR.inkSoft, italic: true,
        align: "left", valign: "top", margin: 0,
      });
    }
  }

  bigStat(s, 0.4,  2.10, 4.55, 1.35, "258,645",  "fault trials",        "across 15 experiments, deterministic seeds");
  bigStat(s, 5.05, 2.10, 4.55, 1.35, "0 of 10,300", "false positives on clean trials", "Wilson upper bound 3.7 × 10⁻⁴");
  bigStat(s, 0.4,  3.55, 2.95, 1.35, "100%",     "five algebraic frontends",          "linear, polynomial, PWL, NN, nonlinear", COLOR.forest);
  bigStat(s, 3.45, 3.55, 2.95, 1.35, "83.2%",    "statistical frontend",              "structural noise-floor ceiling", COLOR.brick);
  bigStat(s, 6.50, 3.55, 3.10, 1.35, "63%",      "energy saved vs always-on TMR",     "modelled 24-month Europa Clipper", COLOR.gold);

  s.addText("Detection at 100% on five frontends is striking, but the cleanest single result is the zero false positives across 10,300 fault-free runs — the framework does not cry wolf.", {
    x: 0.4, y: 4.98, w: 9.2, h: 0.30,
    fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: COLOR.inkSoft, margin: 0,
  });

  footer(s, 9, TOTAL);

  // ===========================================================
  // SLIDE 10 — COMMON-MODE
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Common-mode resilience across the full contamination range", "where TMR fails and we don't", 0.50);

  s.addText("The single sweep that most distinguishes the framework: vary the fraction of faults that are common-mode (correlated across replicas) from 0 to 100% at a fixed per-operation fault rate. TMR's failure rate climbs with correlation. The sheaf's stays at zero across 45,000 trials.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.70,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const cmImg = path.join(FIG_PNG_DIR, "fig08_common_mode.png");
  if (fs.existsSync(cmImg)) {
    s.addImage({ path: cmImg, x: 0.4, y: 2.05, w: 6.0, h: 2.4 });
  }
  const tk = [
    { big: "0%",      label: "sheaf failure across 45,000 trials" },
    { big: "11.3%",   label: "TMR failure at 100% common-mode" },
    { big: "1–10%",   label: "common-mode in heavy-ion data" },
  ];
  let ty = 2.05;
  for (const t of tk) {
    s.addShape("rect", { x: 6.6, y: ty, w: 3.1, h: 0.75,
        fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addShape("rect", { x: 6.6, y: ty, w: 0.08, h: 0.75,
        fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 } });
    s.addText(t.big, {
      x: 6.78, y: ty + 0.06, w: 1.4, h: 0.60,
      fontFace: FONT_HEAD, fontSize: 22, bold: true, color: COLOR.navy,
      valign: "middle", margin: 0,
    });
    s.addText(t.label, {
      x: 8.05, y: ty + 0.06, w: 1.55, h: 0.65,
      fontFace: FONT_BODY, fontSize: 9.5, color: COLOR.ink, valign: "middle", margin: 0,
    });
    ty += 0.85;
  }

  s.addText("The structural reason: TMR votes on identical replicas, so correlated errors are unanimous. The sheaf checks algebraic relations between distinct state variables — correlated noise produces a non-zero syndrome regardless of how the underlying error was distributed.", {
    x: 0.4, y: 4.60, w: 9.2, h: 0.50,
    fontFace: FONT_BODY, fontSize: 10.5, italic: true, color: COLOR.ink, margin: 0,
  });

  footer(s, 10, TOTAL);

  // ===========================================================
  // SLIDE 11 — HONEST NEGATIVES
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Where the framework fails or underperforms", "honest limits", 0.50);

  s.addText("Three regimes where the headline numbers degrade, and one synthetic-program edge case. None of these were buried; surfacing them is necessary to make the strong claims credible.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.50,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const neg = [
    {
      title: "Statistical frontend ceiling",
      stat: "17%",
      sub: "miss rate at deployment scale",
      text: "PCA-derived invariants have a numerical floor near 10⁻². Sub-floor faults are absorbed.",
    },
    {
      title: "Multi-fault degradation",
      stat: "24%",
      sub: "recovery at k = 20 faults",
      text: "OMP recovery guarantee is k ≲ √n. At k=15 we hit 53%, at k=20 we hit 24%.",
    },
    {
      title: "Adversarial LDPC margin",
      stat: "2.1 pp",
      sub: "lead over best random LDPC",
      text: "Best of 20 random binary LDPC codes catches 97.9% of vertex-permutation faults.",
    },
  ];
  let nx = 0.4;
  const ncW = 3.05, ncH = 2.10, ncG = 0.15;
  for (const n of neg) {
    s.addShape("rect", { x: nx, y: 1.95, w: ncW, h: ncH,
        fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addShape("rect", { x: nx, y: 1.95, w: 0.08, h: ncH,
        fill: { color: COLOR.brick }, line: { color: COLOR.brick, width: 0 } });
    s.addText(n.title, {
      x: nx + 0.2, y: 2.03, w: ncW - 0.3, h: 0.30,
      fontFace: FONT_HEAD, fontSize: 11.5, bold: true, color: COLOR.ink, margin: 0,
    });
    s.addText(n.stat, {
      x: nx + 0.2, y: 2.35, w: ncW - 0.3, h: 0.70,
      fontFace: FONT_HEAD, fontSize: 32, bold: true, color: COLOR.brick, margin: 0,
    });
    s.addText(n.sub, {
      x: nx + 0.2, y: 3.00, w: ncW - 0.3, h: 0.25,
      fontFace: FONT_BODY, fontSize: 9.5, italic: true, color: COLOR.inkSoft, margin: 0,
    });
    s.addText(n.text, {
      x: nx + 0.2, y: 3.25, w: ncW - 0.3, h: 0.75,
      fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, margin: 0,
    });
    nx += ncW + ncG;
  }

  s.addShape("rect", { x: 0.4, y: 4.25, w: 9.2, h: 0.85,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addShape("rect", { x: 0.4, y: 4.25, w: 0.08, h: 0.85,
      fill: { color: COLOR.brick }, line: { color: COLOR.brick, width: 0 } });
  s.addText("Plus: 2.3% of randomly generated linear programs fail the structured-map rank lemma across 26,153 tested vertices. Real flight code with continuous sensor calibrations passes the lemma everywhere; only pathological synthetic constants produce the failure. The almost-sure qualifier in the proof is doing real work.", {
    x: 0.6, y: 4.33, w: 8.9, h: 0.70,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, italic: true, margin: 0,
  });

  footer(s, 11, TOTAL);

  // ===========================================================
  // SLIDE 12 — COMPOSED FRONTENDS
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Composing two frontends on one task: a quaternion attitude controller", "compositional protection", 0.50);

  s.addText("A spacecraft attitude controller has both a linear part (a Kalman update on body rates) and a polynomial part (the quaternion norm constraint ‖q‖² = 1). The framework lets us protect the same task three ways: with the linear frontend alone, the polynomial frontend alone, or by stacking both via block-diagonal parity-check matrices. Four fault scenarios were tested at 1,000 trials each.", {
    x: 0.4, y: 1.25, w: 9.2, h: 1.0,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const composedData = [
    { name: "Linear only",       labels: ["Quat bit-flip", "Rate bit-flip", "Quat swap", "Multi-bit burst"], values: [78.2, 66.7, 100.0, 97.5] },
    { name: "Polynomial only",   labels: ["Quat bit-flip", "Rate bit-flip", "Quat swap", "Multi-bit burst"], values: [73.9,  0.0, 100.0, 68.9] },
    { name: "Both composed",     labels: ["Quat bit-flip", "Rate bit-flip", "Quat swap", "Multi-bit burst"], values: [79.3, 66.7, 100.0, 99.5] },
  ];
  s.addChart(pres.charts.BAR, composedData, {
    x: 0.4, y: 2.35, w: 6.4, h: 2.7,
    barDir: "col", barGrouping: "clustered",
    chartColors: [COLOR.navy, COLOR.gold, COLOR.forest],
    chartArea: { fill: { color: "FFFFFF" } },
    plotArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: COLOR.ink, catAxisLabelFontSize: 9,
    valAxisLabelColor: COLOR.inkSoft, valAxisLabelFontSize: 9,
    valGridLine: { color: "E8E2D5", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelFontSize: 8, dataLabelColor: COLOR.ink,
    dataLabelFormatCode: "0",
    valAxisMaxVal: 110, valAxisMinVal: 0,
    showLegend: true, legendPos: "b", legendFontSize: 9, legendColor: COLOR.ink,
  });

  s.addShape("rect", { x: 6.95, y: 2.35, w: 2.75, h: 2.7,
      fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
  s.addShape("rect", { x: 6.95, y: 2.35, w: 0.08, h: 2.7,
      fill: { color: COLOR.gold }, line: { color: COLOR.gold, width: 0 } });
  s.addText("What composition buys", {
    x: 7.10, y: 2.45, w: 2.55, h: 0.28,
    fontFace: FONT_HEAD, fontSize: 11, bold: true, color: COLOR.navy, margin: 0,
  });
  s.addText([
    { text: "Burst scenario:", options: { bold: true, breakLine: true } },
    { text: "97.5% → 99.5%", options: { breakLine: true } },
    { text: "Faults that escape one frontend are caught by the other.", options: { italic: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Rate-flip scenario:", options: { bold: true, breakLine: true } },
    { text: "Polynomial frontend is blind by design (0%) — it doesn't see body rates. Composed regime recovers the linear regime's 66.7%.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Cost:", options: { bold: true, breakLine: false } },
    { text: " block-diagonal stack of two H matrices. No new algorithm." },
  ], {
    x: 7.10, y: 2.78, w: 2.55, h: 2.25,
    fontFace: FONT_BODY, fontSize: 9, color: COLOR.ink, margin: 0,
  });

  footer(s, 12, TOTAL);

  // ===========================================================
  // SLIDE 13 — RUNTIME & MEMORY
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Runtime and memory: microseconds per check, kilobytes per matrix", "engineering tractability", 0.50);

  s.addText("Two measurements taken on this x86-64 Linux container. Memory: the sparse parity-check matrix in CSR format. Latency: median wall-clock time for one syndrome check and for one OMP recovery (single-fault and 5-fault), across 20 trials per frontend.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.65,
    fontFace: FONT_BODY, fontSize: 10.5, color: COLOR.ink, margin: 0,
  });

  s.addChart(pres.charts.BAR, [{
    name: "Sparse CSR memory (KB)",
    labels: ["Linear", "Polynomial", "PWL", "Neural net", "Statistical", "Nonlinear"],
    values: [4.0, 5.9, 0.8, 2.0, 0.9, 10.8],
  }], {
    x: 0.4, y: 1.95, w: 4.5, h: 2.2,
    barDir: "col",
    chartColors: [COLOR.navy],
    chartArea: { fill: { color: "FFFFFF" } }, plotArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: COLOR.ink, catAxisLabelFontSize: 8.5,
    valAxisLabelColor: COLOR.inkSoft, valAxisLabelFontSize: 8.5,
    valGridLine: { color: "E8E2D5", size: 0.5 }, catGridLine: { style: "none" },
    showValue: true, dataLabelFontSize: 8, dataLabelColor: COLOR.ink, dataLabelFormatCode: "0.0",
    showLegend: false,
    showTitle: true, title: "Parity-check matrix footprint (KB)",
    titleFontSize: 11, titleColor: COLOR.navy, titleFontFace: FONT_HEAD,
  });

  s.addChart(pres.charts.BAR, [
    { name: "Syndrome check (µs)", labels: ["Linear","Polynomial","PWL","Neural net","Statistical","Nonlinear"], values: [3.5,4.2,2.9,3.2,3.4,5.5] },
    { name: "OMP k=1 (µs)",        labels: ["Linear","Polynomial","PWL","Neural net","Statistical","Nonlinear"], values: [63,221,44,47,39,659] },
    { name: "OMP k=5 (µs)",        labels: ["Linear","Polynomial","PWL","Neural net","Statistical","Nonlinear"], values: [181,452,143,143,121,1206] },
  ], {
    x: 5.0, y: 1.95, w: 4.7, h: 2.2,
    barDir: "col", barGrouping: "clustered",
    chartColors: [COLOR.forest, COLOR.navy, COLOR.brick],
    chartArea: { fill: { color: "FFFFFF" } }, plotArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: COLOR.ink, catAxisLabelFontSize: 8.5,
    valAxisLabelColor: COLOR.inkSoft, valAxisLabelFontSize: 8.5,
    valGridLine: { color: "E8E2D5", size: 0.5 }, catGridLine: { style: "none" },
    valAxisLogScaleBase: 10,
    showLegend: true, legendPos: "b", legendFontSize: 8.5, legendColor: COLOR.ink,
    showTitle: true, title: "Per-call latency (µs, log scale)",
    titleFontSize: 11, titleColor: COLOR.navy, titleFontFace: FONT_HEAD,
  });

  s.addShape("rect", { x: 0.4, y: 4.30, w: 9.3, h: 0.85,
      fill: { color: "FFFFFF" }, line: { color: COLOR.brick, width: 1 } });
  s.addShape("rect", { x: 0.4, y: 4.30, w: 0.08, h: 0.85,
      fill: { color: COLOR.brick }, line: { color: COLOR.brick, width: 0 } });
  s.addImage({ data: iconWarn, x: 0.6, y: 4.45, w: 0.30, h: 0.30 });
  s.addText("Caveat: x86-64 Linux, not RAD750.", {
    x: 0.95, y: 4.38, w: 4.5, h: 0.30,
    fontFace: FONT_HEAD, fontSize: 11, bold: true, color: COLOR.brick, margin: 0,
  });
  s.addText("Absolute numbers on flight hardware will differ. The portable claim is the scaling: linear-in-nnz syndrome check, O(n^1.5) OMP on dense H, sparse-CSR footprint dominated by non-zero count. All three trends persist regardless of processor.", {
    x: 0.95, y: 4.65, w: 8.65, h: 0.45,
    fontFace: FONT_BODY, fontSize: 9.5, color: COLOR.ink, italic: true, margin: 0,
  });

  footer(s, 13, TOTAL);

  // ===========================================================
  // SLIDE 14 — MISSION ENERGY
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.navyDark };

  s.addText("MISSION-INTEGRATED ENERGY", {
    x: 0.4, y: 0.5, w: 9.2, h: 0.25,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.gold, bold: true, charSpacing: 4, margin: 0,
  });
  s.addText("Adaptive sheaf on a modelled 24-month Europa Clipper profile", {
    x: 0.4, y: 0.80, w: 9.2, h: 0.45,
    fontFace: FONT_HEAD, fontSize: 18, bold: true, color: "FFFFFF", margin: 0,
  });
  hrule(s, 0.4, 1.35, 1.5, COLOR.gold, 1.8);

  s.addText("63%", {
    x: 0.4, y: 1.65, w: 5.0, h: 2.4,
    fontFace: FONT_HEAD, fontSize: 130, bold: true, color: COLOR.gold,
    align: "left", valign: "middle", margin: 0,
  });
  s.addText("energy saved at 97.6% mean coverage", {
    x: 5.4, y: 2.30, w: 4.2, h: 0.75,
    fontFace: FONT_HEAD, fontSize: 15, color: "FFFFFF", bold: true, valign: "middle", margin: 0,
  });
  s.addText("The adaptive strategy switches between low-coverage cruise and high-coverage flyby configurations based on a particle-flux threshold. The 63% number reflects the integrated energy across six Jupiter flybys at months 3, 7, 11, 14, 17, and 20 against always-on TMR.", {
    x: 5.4, y: 3.05, w: 4.2, h: 1.00,
    fontFace: FONT_BODY, fontSize: 10.5, color: "DDD7C6", italic: true, margin: 0,
  });

  hrule(s, 0.4, 4.25, 9.2, COLOR.gold, 0.8);
  const compare = [
    { lbl: "TMR always", val: "72 unit-mo @ 98.9%", c: COLOR.brick },
    { lbl: "Max sheaf",  val: "36 unit-mo @ 99.0%", c: COLOR.goldLite },
    { lbl: "Adaptive (ours)", val: "27 unit-mo @ 97.6%", c: COLOR.gold },
    { lbl: "Min sheaf",  val: "24 unit-mo @ 92.0%", c: "B6BFD0" },
  ];
  let xc = 0.4; const cw = 2.30;
  for (const c of compare) {
    s.addText(c.lbl, { x: xc, y: 4.40, w: cw, h: 0.22,
      fontFace: FONT_BODY, fontSize: 10, color: c.c, bold: true, margin: 0 });
    s.addText(c.val, { x: xc, y: 4.63, w: cw, h: 0.30,
      fontFace: FONT_HEAD, fontSize: 12, color: "FFFFFF", margin: 0 });
    xc += cw + 0.05;
  }
  s.addText("The adaptive strategy beats max-sheaf on energy and TMR on both axes simultaneously.", {
    x: 0.4, y: 5.00, w: 9.2, h: 0.25,
    fontFace: FONT_BODY, fontSize: 10, italic: true, color: "B6BFD0", margin: 0,
  });

  s.addText("14 / 16", {
    x: 8.6, y: 5.30, w: 1.0, h: 0.2,
    fontFace: FONT_BODY, fontSize: 9, color: "8A98B0", align: "right", margin: 0,
  });

  // ===========================================================
  // SLIDE 15 — OTHER APPLICATIONS
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.cream };
  slideTitle(s, "Where else the same edge applies", "beyond spacecraft", 0.50);

  s.addText("The framework's specific advantages — semantic invariants instead of bit-level parity, immunity to common-mode contamination, frontend extensibility — match the engineering needs of three other domains where similar pressures apply.", {
    x: 0.4, y: 1.25, w: 9.2, h: 0.65,
    fontFace: FONT_BODY, fontSize: 11, color: COLOR.ink, margin: 0,
  });

  const apps = [
    {
      icon: wCar, name: "Autonomous-vehicle control",
      body: "Electronic stability, adaptive cruise, lane-keeping loops face transient faults from electromagnetic interference and aging-induced bit upsets. The linear and polynomial frontends cover the typical control-loop invariants; common-mode resilience addresses the correlated failures that affect physically adjacent transistors.",
    },
    {
      icon: wHeart, name: "Medical device firmware",
      body: "Insulin pumps and cardiac pacemakers carry regulatory-critical dose-calculation and pacing-decision logic. The statistical frontend's principal-component approach naturally covers the patient-specific calibration curves these devices maintain across their service life.",
    },
    {
      icon: wCog, name: "Industrial process control",
      body: "Nuclear plant programmable logic controllers and refinery distributed control systems carry sensor mass-balance, energy-balance, and thermodynamic-consistency relations — exactly the algebraic invariants the polynomial frontend was built to encode.",
    },
  ];
  let ay = 2.00;
  for (const a of apps) {
    s.addShape("rect", { x: 0.4, y: ay, w: 9.3, h: 1.05,
        fill: { color: "FFFFFF" }, line: { color: COLOR.ruleLite, width: 0.75 } });
    s.addShape("rect", { x: 0.4, y: ay, w: 0.08, h: 1.05,
        fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 } });
    s.addShape("ellipse", { x: 0.65, y: ay + 0.25, w: 0.55, h: 0.55,
        fill: { color: COLOR.navy }, line: { color: COLOR.navy, width: 0 } });
    s.addImage({ data: a.icon, x: 0.76, y: ay + 0.36, w: 0.33, h: 0.33 });
    s.addText(a.name, {
      x: 1.40, y: ay + 0.10, w: 8.0, h: 0.30,
      fontFace: FONT_HEAD, fontSize: 12.5, bold: true, color: COLOR.navy, margin: 0,
    });
    s.addText(a.body, {
      x: 1.40, y: ay + 0.38, w: 8.1, h: 0.65,
      fontFace: FONT_BODY, fontSize: 10, color: COLOR.ink, margin: 0,
    });
    ay += 1.10;
  }

  footer(s, 15, TOTAL);

  // ===========================================================
  // SLIDE 16 — CONCLUSION
  // ===========================================================
  s = pres.addSlide();
  s.background = { color: COLOR.navyDark };

  s.addText("CONCLUSION", {
    x: 0.6, y: 0.55, w: 9.0, h: 0.25,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.gold, bold: true, charSpacing: 4, margin: 0,
  });
  s.addText("What the framework establishes", {
    x: 0.6, y: 0.85, w: 9.0, h: 0.45,
    fontFace: FONT_HEAD, fontSize: 22, bold: true, italic: true, color: "FFFFFF", margin: 0,
  });
  s.addText("A single mathematical object — the cellular sheaf — recovers Hamming/SECDED, ABFT, and SWIFT as special cases. It dominates binary low-density parity-check codes on the same Tanner graph on a structurally identifiable fault class. It delivers 63% mission-integrated energy savings against always-on TMR while preserving immunity to common-mode contamination.", {
    x: 0.6, y: 1.50, w: 9.0, h: 1.40,
    fontFace: FONT_BODY, fontSize: 12.5, color: "DDD7C6", margin: 0,
  });

  hrule(s, 0.6, 3.10, 1.5, COLOR.gold, 1.8);

  s.addText("THE NEXT STEPS", {
    x: 0.6, y: 3.25, w: 9.0, h: 0.25,
    fontFace: FONT_BODY, fontSize: 10, color: COLOR.gold, bold: true, charSpacing: 4, margin: 0,
  });
  const next = [
    "Heavy-ion beam test on a radiation-hardened processor — replace modelled fault distributions with measured ones.",
    "Extend the Sheaf-LDPC separation from within-vertex to cross-vertex permutations.",
    "Apply the framework to autonomous-vehicle and medical-device firmware as a compiler pass.",
  ];
  s.addText(next.map((t, i) => ({ text: t, options: { bullet: { code: "25B8" }, breakLine: i < next.length - 1, color: "DDD7C6" } })), {
    x: 0.6, y: 3.55, w: 9.0, h: 1.30,
    fontFace: FONT_BODY, fontSize: 11.5, color: "DDD7C6", paraSpaceAfter: 5, margin: 0,
  });

  hrule(s, 0.6, 4.90, 9.0, COLOR.navyLight, 0.5);
  s.addImage({ data: iconGit, x: 0.6, y: 5.05, w: 0.20, h: 0.20 });
  s.addText(GITHUB_URL, {
    x: 0.85, y: 5.04, w: 5.5, h: 0.22,
    fontFace: "Consolas", fontSize: 11, color: COLOR.gold, margin: 0,
  });
  s.addText("Vaibhav Shakya  ·  Jayshree Periwal International School  ·  GRC 2026", {
    x: 0.6, y: 5.30, w: 9.0, h: 0.22,
    fontFace: FONT_BODY, fontSize: 10, color: "B6BFD0", italic: true, margin: 0,
  });

  // Save
  const out = path.join(__dirname, "Leray_Coboundary_at_Altitude_GRC.pptx");
  await pres.writeFile({ fileName: out });
  console.log("Wrote:", out);
}

main().catch(e => { console.error(e); process.exit(1); });
