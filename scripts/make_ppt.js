const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");

// --- Icon helpers -----------------------------------------------------------
const {
  FaDrone, FaMapMarkerAlt, FaExclamationTriangle, FaCheckCircle,
  FaCamera, FaBrain, FaLayerGroup, FaCrosshairs, FaRobot,
  FaShieldAlt, FaSync, FaTimesCircle, FaArrowRight, FaArrowDown,
  FaSearch, FaChartBar, FaCog, FaPlane, FaSatellite
} = require("react-icons/fa");
const { MdFlightLand, MdWarning, MdLocationOn, MdSpeed } = require("react-icons/md");

async function iconPng(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// --- Colours ----------------------------------------------------------------
const C = {
  navy:   "0D1B3E",
  blue:   "1E40AF",
  blueL:  "3B82F6",
  green:  "16A34A",
  greenL: "22C55E",
  red:    "DC2626",
  redL:   "EF4444",
  amber:  "D97706",
  amberL: "F59E0B",
  white:  "FFFFFF",
  offW:   "F8FAFC",
  light:  "EFF6FF",
  grey:   "64748B",
  greyL:  "CBD5E1",
  dark:   "1E293B",
  teal:   "0D9488",
};

function rect(slide, x, y, w, h, fill, line, opts = {}) {
  slide.addShape("rect", { x, y, w, h, fill: { color: fill }, line: line ? { color: line, width: 1.5 } : { type: "none" }, ...opts });
}
function roundRect(slide, x, y, w, h, fill, radius = 0.1, opts = {}) {
  slide.addShape("roundRect", { x, y, w, h, fill: { color: fill }, rectRadius: radius, line: { type: "none" }, ...opts });
}
function txt(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { x, y, w, h, margin: 0, ...opts });
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";

  // Minimal version with same style + key structure from your content
  // (kept compact so it runs cleanly in this environment).

  // SLIDE 1 – Title
  {
    const sl = pres.addSlide();
    sl.background = { color: C.navy };
    rect(sl, 0, 0, 0.12, 5.625, C.blueL);
    txt(sl, "Infrastructure-Free\nAutonomous UAV Landing", 0.4, 0.45, 8.5, 2.1, {
      fontSize: 34, bold: true, color: C.white, fontFace: "Calibri"
    });
    txt(sl, "in Unknown Terrain via Semantic Detection,\nEvidence-Grid Fusion, and Abort-Capable State Control",
      0.4, 2.4, 9, 0.9, { fontSize: 15, color: C.greyL, fontFace: "Calibri", italic: true });
    rect(sl, 0.4, 3.25, 9.2, 0.03, C.blueL);
    txt(sl, "Research Presentation  ·  BITS Pilani Dubai Campus", 0.4, 3.42, 9, 0.45, {
      fontSize: 13, color: C.greyL, fontFace: "Calibri"
    });
  }

  // SLIDE 2 – Problem
  {
    const sl = pres.addSlide();
    sl.background = { color: C.offW };
    txt(sl, "Where can a UAV land safely…", 0.5, 0.2, 9, 0.7, {
      fontSize: 28, bold: true, color: C.dark, fontFace: "Calibri", align: "center"
    });
    txt(sl, "…when there are NO maps, NO markers, and NO prepared infrastructure?",
      0.5, 0.85, 9, 0.6, { fontSize: 16, color: C.grey, fontFace: "Calibri", align: "center", italic: true });
    const droneIcon = await iconPng(FaPlane, "#1E40AF", 256);
    sl.addImage({ data: droneIcon, x: 4.4, y: 1.55, w: 0.7, h: 0.7 });
  }

  // SLIDE 3 – Pipeline
  {
    const sl = pres.addSlide();
    sl.background = { color: C.navy };
    txt(sl, "End-to-End Pipeline", 0.5, 0.1, 9, 0.5, {
      fontSize: 24, bold: true, color: C.white, fontFace: "Calibri", align: "center"
    });
    const steps = [
      { label: "Camera", col: "1E3A5F", tcol: C.greyL },
      { label: "YOLO", col: C.blueL, tcol: C.white },
      { label: "Ontology", col: "7C3AED", tcol: C.white },
      { label: "Fusion", col: C.teal, tcol: C.white },
      { label: "Zone", col: C.green, tcol: C.white },
      { label: "FSM", col: C.amberL, tcol: C.dark },
      { label: "Motion", col: C.redL, tcol: C.white },
    ];
    const boxW = 1.2, boxH = 1.1, startX = 0.25, arrowLen = 0.4, yPos = 1.75;
    steps.forEach((s, i) => {
      const bx = startX + i * (boxW + arrowLen);
      roundRect(sl, bx, yPos, boxW, boxH, s.col, 0.12);
      txt(sl, s.label, bx, yPos, boxW, boxH, {
        fontSize: 10, bold: true, color: s.tcol, align: "center", valign: "middle", fontFace: "Calibri"
      });
      if (i < steps.length - 1) {
        sl.addShape("line", {
          x: bx + boxW, y: yPos + boxH / 2, w: arrowLen, h: 0,
          line: { color: C.greyL, width: 2 }, lineHead: "arrow", lineTail: "none"
        });
      }
    });
  }

  // SLIDE 4 – Results
  {
    const sl = pres.addSlide();
    sl.background = { color: C.navy };
    txt(sl, "Experimental Results — 5 Terrain Seeds", 0.5, 0.12, 9, 0.55, {
      fontSize: 24, bold: true, color: C.white, fontFace: "Calibri", align: "center"
    });
    const stats = [
      { val: "100%", label: "Landing\nSuccess Rate", col: C.greenL },
      { val: "35.4s", label: "Mean Time\nto Land", col: C.amberL },
      { val: "17.3s", label: "Mean Grid\nConvergence", col: C.blueL },
      { val: "<33ms", label: "Abort Detection\nLatency", col: C.redL },
    ];
    stats.forEach((s, i) => {
      roundRect(sl, 0.3 + i * 2.38, 0.8, 2.1, 2.2, "1A2744", 0.18);
      txt(sl, s.val, 0.3 + i * 2.38, 0.92, 2.1, 1.1, {
        fontSize: 38, bold: true, color: s.col, align: "center", fontFace: "Calibri"
      });
      txt(sl, s.label, 0.3 + i * 2.38, 2.0, 2.1, 0.8, {
        fontSize: 11, color: C.greyL, align: "center", fontFace: "Calibri"
      });
    });
  }

  // SLIDE 5 – Conclusion
  {
    const sl = pres.addSlide();
    sl.background = { color: C.navy };
    rect(sl, 0, 0, 10, 1.1, C.green);
    txt(sl, "Conclusion", 0.5, 0.18, 9, 0.72, {
      fontSize: 34, bold: true, color: C.white, fontFace: "Calibri", align: "center"
    });
    const takeaways = [
      { text: "100% landing success across 5 terrain seeds", col: C.greenL },
      { text: "Sub-frame abort detection latency", col: C.amberL },
      { text: "Lightweight YOLO + decaying evidence grid", col: C.blueL },
      { text: "PyBullet ? ROS 2/Gazebo/PX4 deployment path", col: "A78BFA" },
    ];
    takeaways.forEach((t, i) => {
      roundRect(sl, 0.3, 1.5 + i * 0.78, 9.4, 0.65, "1A2744", 0.09);
      rect(sl, 0.3, 1.5 + i * 0.78, 0.12, 0.65, t.col);
      txt(sl, "?  " + t.text, 0.6, 1.5 + i * 0.78, 8.8, 0.65, {
        fontSize: 13, color: t.col, fontFace: "Calibri", valign: "middle", bold: true
      });
    });
  }

  const outPath = path.join(process.cwd(), "YOLOSLAM_Presentation.pptx");
  await pres.writeFile({ fileName: outPath });
  console.log("Done! Wrote:", outPath);
})();
