import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableModel;
import java.awt.geom.*;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

/** Sauvegarde / Ouverture de projet .mhd.json */
public final class ProjectIO {

    // ====== PUBLIC: Appelle juste ces 2 méthodes depuis ton UI ======
    public static boolean saveWithChooser(JFrame parent, File currentFile) {
        try {
            File f = (currentFile != null) ? currentFile : chooseSaveFile(parent);
            if (f == null) return false;
            Document doc = Document.fromGlobals();
            MAPPER.writerWithDefaultPrettyPrinter().writeValue(f, doc);
            return true;
        } catch (Exception ex) {
            showErr(parent, "Erreur d'enregistrement", ex);
            return false;
        }
    }

    public static File openWithChooser(JFrame parent) {
        try {
            JFileChooser ch = new JFileChooser();
            ch.setDialogTitle("Ouvrir un projet");
            ch.setFileFilter(new FileNameExtensionFilter("Projet MHD (*.mhd.json)", "json", "mhd.json"));
            if (ch.showOpenDialog(parent) != JFileChooser.APPROVE_OPTION) return null;
            File f = ch.getSelectedFile();
            byte[] bytes = Files.readAllBytes(f.toPath());
            Document doc = MAPPER.readValue(new String(bytes, StandardCharsets.UTF_8), Document.class);
            doc.applyToGlobals();
            return f;
        } catch (Exception ex) {
            showErr(parent, "Erreur d'ouverture", ex);
            return null;
        }
    }

    // ====== MAPPER JSON ======
    private static final ObjectMapper MAPPER = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT)
            .setSerializationInclusion(JsonInclude.Include.NON_NULL);

    private static void showErr(java.awt.Component parent, String title, Exception ex){
        ex.printStackTrace();
        JOptionPane.showMessageDialog(parent, ex.toString(), title, JOptionPane.ERROR_MESSAGE);
    }

    private static File chooseSaveFile(JFrame parent){
        JFileChooser ch = new JFileChooser();
        ch.setDialogTitle("Enregistrer le projet");
        ch.setFileFilter(new FileNameExtensionFilter("Projet MHD (*.mhd.json)", "json", "mhd.json"));
        if (ch.showSaveDialog(parent) != JFileChooser.APPROVE_OPTION) return null;
        File f = ch.getSelectedFile();
        String name = f.getName().toLowerCase(Locale.ROOT);
        if (!name.endsWith(".mhd.json")) {
            f = new File(f.getParentFile(), f.getName() + ".mhd.json");
        }
        return f;
    }

    // ========================================================================
    // =======================  DOCUMENT (DTO JSON)  ==========================
    // ========================================================================
    public static final class Document {
        public String version = "0.1.0";
        public Meta meta = new Meta();

        public List<Param> parameters = new ArrayList<>();

        public Geometry geometry = new Geometry();
        public Domains  domains  = new Domains();
        public Materials materials = new Materials();
        // Tu ajouteras plus tard: boundaries, initial, mesh, solver, etc.

        public static Document fromGlobals() {
            Document d = new Document();
            d.meta.modified = new Date().toString();

            // --- Paramètres
            if (Globals.paramsModel != null) {
                DefaultTableModel m = Globals.paramsModel;
                for (int r=0; r<m.getRowCount(); r++){
                    String n = get(m, r, 0), u = get(m, r, 1), v = get(m, r, 2), desc = get(m, r, 3);
                    if (n != null && !n.isBlank()) d.parameters.add(new Param(n,u,v,desc));
                }
            }

            // --- Géométrie
            for (ShapeItem it : Globals.shapes){
                DocShape ds = new DocShape();
                ds.id = it.id;
                ds.kind = it.kind.name();
                switch (it.kind){
                    case RECT -> {
                        Rectangle2D b = it.shape.getBounds2D();
                        Rect rr = new Rect();
                        rr.x = b.getX(); rr.y=b.getY(); rr.w=b.getWidth(); rr.h=b.getHeight();
                        ds.rect = rr;
                    }
                    case OVAL -> {
                        Ellipse2D e = (Ellipse2D) it.shape.getBounds2D();
                        // On récupère centre/rayons à partir de la bbox (suffisant pour tes ajouts)
                        Oval ov = new Oval();
                        ov.xc = e.getCenterX(); ov.yc = e.getCenterY();
                        ov.rx = e.getWidth()/2.0; ov.ry = e.getHeight()/2.0;
                        ds.oval = ov;
                    }
                    case LINE -> {
                        Line2D l = (Line2D) it.shape;
                        Line ln = new Line();
                        ln.x1=l.getX1(); ln.y1=l.getY1(); ln.x2=l.getX2(); ln.y2=l.getY2();
                        ds.line = ln;
                    }
                    case POLY -> {
                        // Optionnel: si tu utilises buildClosedPathFromSelectedLines() qui renvoie Path2D
                        // tu peux approximer en liste de points. Pour l’instant on ignore si tu n’en crées pas encore.
                        ds.poly = pathToPoints(it.shape);
                    }
                }
                d.geometry.shapes.add(ds);
            }
            // edges
            d.geometry.shapeEdgeLines = new LinkedHashMap<>();
            for (var e : Globals.shapeEdgeLines.entrySet()){
                d.geometry.shapeEdgeLines.put(e.getKey(), new ArrayList<>(e.getValue()));
            }

            // --- Domaines
            if (Globals.domainsModel != null) {
                DefaultTableModel m = Globals.domainsModel;
                for (int r=0; r<m.getRowCount(); r++){
                    String name = get(m, r, 0);
                    String descr= get(m, r, 1);
                    String figs = get(m, r, 2);
                    if (name!=null && !name.isBlank()){
                        d.domains.table.add(new DomainRow(name, descr, figs));
                    }
                }
            }
            // Edits (source de vérité pour reconstruire les areas)
            d.domains.edits = new LinkedHashMap<>();
            for (var e : Globals.domainEdits.entrySet()){
                List<DomainEditRow> L = new ArrayList<>();
                for (Globals.DomainEdit de : e.getValue()) {
                    L.add(new DomainEditRow(de.shapeId, de.op.name()));
                }
                d.domains.edits.put(e.getKey(), L);
            }

            // --- Matériaux : noms
            for (int i=0;i<Globals.materialsModel.getSize();i++){
                d.materials.list.add(Globals.materialsModel.getElementAt(i));
            }
            // Assignements domaines
            d.materials.assignments = new LinkedHashMap<>();
            for (var e : Globals.materialDomains.entrySet()){
                d.materials.assignments.put(e.getKey(), new ArrayList<>(e.getValue()));
            }
            // Propriétaires
            d.materials.owners = new LinkedHashMap<>(Globals.domainOwner);

            // Variables par matériau
            d.materials.vars = new LinkedHashMap<>();
            for (var e : Globals.matVars.entrySet()){
                String mat = e.getKey();
                Map<String, DocVarSpec> vars = new LinkedHashMap<>();
                for (var vv : e.getValue().entrySet()){
                    Globals.VarSpec s = vv.getValue();
                    DocVarSpec dv = new DocVarSpec();
                    dv.useFunc = s.useFunc; dv.funcExpr = s.funcExpr; dv.csvPath = s.csvPath;
                    dv.useT = s.useT; dv.useP = s.useP; dv.useConc = s.useConc;
                    vars.put(vv.getKey(), dv);
                }
                d.materials.vars.put(mat, vars);
            }

            // Réactions
            d.materials.reactions = new LinkedHashMap<>();
            for (var e : Globals.matReactions.entrySet()){
                List<DocReaction> list = new ArrayList<>();
                for (Globals.ReactionSpec r : e.getValue()){
                    DocReaction rr = new DocReaction();
                    rr.kExpr=r.kExpr; rr.energyMeV=r.energyMeV; rr.products=r.products; rr.reactants=r.reactants;
                    list.add(rr);
                }
                d.materials.reactions.put(e.getKey(), list);
            }

            return d;
        }

        public void applyToGlobals() {
            // Reset structures
            Globals.shapes.clear();
            Globals.shapeEdgeLines.clear();
            Globals.domainAreas.clear();
            Globals.domainEdits.clear();
            Globals.materialDomains.clear();
            Globals.domainOwner.clear();
            Globals.matVars.clear();
            Globals.matReactions.clear();
            Globals.materialsModel.removeAllElements();
            Globals.rectCount = Globals.ovalCount = Globals.lineCount = Globals.polyCount = 0;

            // --- Params
            if (Globals.paramsModel != null) {
                var m = Globals.paramsModel;
                // vide le modèle
                while (m.getRowCount()>0) m.removeRow(0);
                for (Param p : parameters){
                    m.addRow(new Object[]{p.name, p.unit, p.value, p.desc});
                }
            }

            // --- Geometry (recrée les shapes + compteurs)
            Map<String, ShapeItem> id2shape = new LinkedHashMap<>();
            for (DocShape ds : geometry.shapes){
                ShapeItem si = null;
                ShapeKind kind = ShapeKind.valueOf(ds.kind);
                switch (kind){
                    case RECT -> {
                        Rect r = ds.rect;
                        var rect = new Rectangle2D.Double(r.x, r.y, r.w, r.h);
                        si = new ShapeItem(ds.id, ShapeKind.RECT, rect);
                        Globals.rectCount = Math.max(Globals.rectCount, parseTrailingInt(ds.id, 'r'));
                    }
                    case OVAL -> {
                        Oval o = ds.oval;
                        var el = new Ellipse2D.Double(o.xc - o.rx, o.yc - o.ry, 2*o.rx, 2*o.ry);
                        si = new ShapeItem(ds.id, ShapeKind.OVAL, el);
                        Globals.ovalCount = Math.max(Globals.ovalCount, parseTrailingInt(ds.id, "ov"));
                    }
                    case LINE -> {
                        Line l = ds.line;
                        var ln = new Line2D.Double(l.x1,l.y1,l.x2,l.y2);
                        si = new ShapeItem(ds.id, ShapeKind.LINE, ln);
                        Globals.lineCount = Math.max(Globals.lineCount, parseTrailingInt(ds.id, 'l'));
                    }
                    case POLY -> {
                        Path2D path = pointsToPath(ds.poly);
                        si = new ShapeItem(ds.id, ShapeKind.POLY, path);
                        Globals.polyCount = Math.max(Globals.polyCount, parseTrailingInt(ds.id, "poly"));
                    }
                }
                if (si != null){
                    Globals.shapes.add(si);
                    id2shape.put(ds.id, si);
                }
            }
            // edges
            if (geometry.shapeEdgeLines != null) {
                for (var e : geometry.shapeEdgeLines.entrySet()){
                    Globals.shapeEdgeLines.put(e.getKey(), new ArrayList<>(e.getValue()));
                }
            }

            // --- Domains
            if (Globals.domainsModel != null && domains.table != null) {
                var m = Globals.domainsModel;
                while (m.getRowCount()>0) m.removeRow(0);
                for (DomainRow row : domains.table){
                    m.addRow(new Object[]{row.name, row.description, row.figures});
                }
            }
            if (domains.edits != null) {
                for (var e : domains.edits.entrySet()){
                    List<Globals.DomainEdit> L = new ArrayList<>();
                    for (DomainEditRow r : e.getValue()){
                        L.add(new Globals.DomainEdit(r.shapeId, Globals.DomOp.valueOf(r.op)));
                    }
                    Globals.domainEdits.put(e.getKey(), L);
                }
                Globals.fireDomainsChanged();
            }

            // --- Materials
            if (materials != null){
                if (materials.list != null)
                    for (String name : materials.list) Globals.ensureMaterial(name);

                if (materials.assignments != null)
                    for (var e : materials.assignments.entrySet())
                        Globals.materialDomains.put(e.getKey(), new LinkedHashSet<>(e.getValue()));

                if (materials.owners != null)
                    Globals.domainOwner.putAll(materials.owners);

                if (materials.vars != null) {
                    for (var e : materials.vars.entrySet()){
                        String mat = e.getKey();
                        for (var v : e.getValue().entrySet()){
                            Globals.VarSpec s = Globals.getVarSpec(mat, v.getKey());
                            DocVarSpec dv = v.getValue();
                            s.useFunc=dv.useFunc; s.funcExpr=dv.funcExpr; s.csvPath=dv.csvPath;
                            s.useT=dv.useT; s.useP=dv.useP; s.useConc=dv.useConc;
                        }
                    }
                }
                if (materials.reactions != null){
                    for (var e : materials.reactions.entrySet()){
                        List<Globals.ReactionSpec> L = Globals.getReactions(e.getKey());
                        L.clear();
                        for (DocReaction r : e.getValue()){
                            Globals.ReactionSpec rr = new Globals.ReactionSpec();
                            rr.kExpr=r.kExpr; rr.energyMeV=r.energyMeV; rr.products=r.products; rr.reactants=r.reactants;
                            L.add(rr);
                        }
                    }
                }
            }

            Globals.repaintAll();
        }

        // ---------------- DTOs ----------------
        public static final class Meta { public String created = new Date().toString(); public String modified = created; }
        public static final class Param {
            public String name, unit, value, desc;
            public Param() {}
            public Param(String n, String u, String v, String d){ name=n; unit=u; value=v; desc=d; }
        }

        public static final class Geometry {
            public List<DocShape> shapes = new ArrayList<>();
            public Map<String, List<String>> shapeEdgeLines = new LinkedHashMap<>();
        }

        public static final class DocShape {
            public String id;
            public String kind; // RECT/OVAL/LINE/POLY
            public Rect rect; public Oval oval; public Line line;
            public List<double[]> poly; // liste de points [x,y]
        }
        public static final class Rect { public double x,y,w,h; }
        public static final class Oval { public double xc,yc,rx,ry; }
        public static final class Line { public double x1,y1,x2,y2; }

        public static final class Domains {
            public List<DomainRow> table = new ArrayList<>();
            public Map<String, List<DomainEditRow>> edits = new LinkedHashMap<>();
        }
        public static final class DomainRow {
            public String name, description, figures;
            public DomainRow() {}
            public DomainRow(String n, String d, String f){ name=n; description=d; figures=f; }
        }
        public static final class DomainEditRow { public String shapeId, op; public DomainEditRow(){} public DomainEditRow(String s, String o){shapeId=s; op=o;} }

        public static final class Materials {
            public List<String> list = new ArrayList<>();
            public Map<String, List<String>> assignments = new LinkedHashMap<>();
            public Map<String, String> owners = new LinkedHashMap<>();
            public Map<String, Map<String, DocVarSpec>> vars = new LinkedHashMap<>();
            public Map<String, List<DocReaction>> reactions = new LinkedHashMap<>();
        }
        public static final class DocVarSpec {
            public boolean useFunc=true; public String funcExpr=""; public String csvPath="";
            public boolean useT=false, useP=false, useConc=false;
        }
        public static final class DocReaction {
            public String kExpr="", energyMeV="", products="", reactants="";
        }

        // ---------------- Utils serialize/parse ----------------
        private static String get(DefaultTableModel m, int r, int c){
            Object v = m.getValueAt(r, c);
            return (v==null) ? "" : v.toString();
        }

        private static int parseTrailingInt(String id, char prefix){
            try {
                if (id!=null && id.length()>=2 && id.charAt(0)==prefix)
                    return Integer.parseInt(id.substring(1));
            } catch (Exception ignore){}
            return 0;
        }
        private static int parseTrailingInt(String id, String prefix){
            try {
                if (id!=null && id.startsWith(prefix))
                    return Integer.parseInt(id.substring(prefix.length()));
            } catch (Exception ignore){}
            return 0;
        }

        private static List<double[]> pathToPoints(java.awt.Shape s){
            List<double[]> pts = new ArrayList<>();
            PathIterator it = s.getPathIterator(null, 0.5);
            double[] buf = new double[6];
            while (!it.isDone()){
                int t = it.currentSegment(buf);
                if (t==PathIterator.SEG_MOVETO || t==PathIterator.SEG_LINETO) {
                    pts.add(new double[]{buf[0], buf[1]});
                }
                it.next();
            }
            return pts.isEmpty()?null:pts;
        }
        private static Path2D pointsToPath(List<double[]> pts){
            if (pts==null || pts.isEmpty()) return null;
            Path2D p = new Path2D.Double();
            p.moveTo(pts.get(0)[0], pts.get(0)[1]);
            for (int i=1;i<pts.size();i++) p.lineTo(pts.get(i)[0], pts.get(i)[1]);
            p.closePath();
            return p;
        }
    }
}
