/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Controller;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.StringReader;
import java.text.DecimalFormat;
import java.util.Random;
import weka.associations.Apriori;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

/**
 *
 * @author Johnny
 */
public class CtlDataMining {

    //definimos el formato para los decimales
    DecimalFormat formato = new DecimalFormat("#.##");

    public String definirEncabezado(Instances data) {
        /*Se define el encabezado del mensaje, teniendo en cuanta el atributo clase*/
        String descripcion = "<b>El atributo clase seleccionado es "
                + data.attribute(data.numAttributes() - 1).name() + "</b>";
        descripcion += " <b>con posibles valores:</b> ";
        /*Se recorren los posibles valores del atributo clase*/
        for (int z = 0; z < data.attribute(data.numAttributes() - 1).numValues(); z++) {
            descripcion += "<b>" + data.attribute(data.numAttributes() - 1).value(z) + "</b> ";
        }

        return descripcion;
    }

    /*Funcion que aplica una red bayesiana a un conjunto de datos 
    recibidos por parametro*/
    public String redBayesiana(Instances data) {
        try {
            //Creamos un clasificador Bayesiano                
            NaiveBayes nb = new NaiveBayes();

            //creamos el clasificador de la redBayesiana 
            nb.buildClassifier(data);

            //Creamos un objeto para la validacion del modelo con redBayesiana
            Evaluation evalB = new Evaluation(data);

            /*Aplicamos el clasificador bayesiano
            hacemos validacion cruzada, de redBayesiana, con 10 campos, 
            y un aleatorio para la semilla, en este caso es 1 para el 
            muestreo de la validacion cruzada (Como ordenar para luego
            partirlo en 10)*/
            evalB.crossValidateModel(nb, data, 10, new Random(1));

            String resBay = "<br><br><b><center>Resultados NaiveBayes</center>"
                    + "<br>========<br>"
                    + "Modelo generado indica los siguientes resultados:"
                    + "<br>========<br></b>";
            //Obtenemos resultados
            resBay = resBay + ("<b>1. Numero de instancias clasificadas:</b> "
                    + (int) evalB.numInstances() + "<br>");
            resBay = resBay + ("<b>2. Porcentaje de instancias correctamente "
                    + "clasificadas:</b> " + formato.format(evalB.pctCorrect())
                    + "%<br>");
            resBay = resBay + ("<b>3. Numero de instancias correctamente "
                    + "clasificadas:</b> " + (int) evalB.correct() + "<br>");
            resBay = resBay + ("<b>4. Porcentaje de instancias incorrectamente "
                    + "clasificadas:</b> " + formato.format(evalB.pctIncorrect())
                    + "%<br>");
            resBay = resBay + ("<b>5. Numero de instancias incorrectamente "
                    + "clasificadas:</b> " + (int) evalB.incorrect() + "<br>");
            resBay = resBay + ("<b>6. Media del error absoluto:</b> "
                    + formato.format(evalB.meanAbsoluteError()) + "%<br>");
            resBay = resBay + ("<b>7. " + evalB.toMatrixString("Matriz de "
                    + "confusion</b>").replace("\n", "<br>"));

            return resBay;

        } catch (Exception e) {
            return "El error es" + e.getMessage();
        }
    }

    /*Funcion que aplica arbol de prediccion a un conjunto de datos 
    recibidos por parametro*/
    public String arbolJ48(Instances data) {
        try {
            // Creamos un clasidicador J48
            J48 j48 = new J48();
            //creamos el clasificador  del J48 con los datos 
            j48.buildClassifier(data);

            //Creamos un objeto para la validacion del modelo con redBayesiana
            Evaluation evalJ48 = new Evaluation(data);

            /*Aplicamos el clasificador J48
            hacemos validacion cruzada, de redBayesiana, con 10 campos, 
            y el aleatorio arrancando desde 1 para la semilla*/
            evalJ48.crossValidateModel(j48, data, 10, new Random(1));
            //Obtenemos resultados
            String resJ48 = "<br><b><center>Resultados Arbol de decision J48"
                    + "</center><br>========<br>Modelo generado indica los "
                    + "siguientes resultados:<br>========<br></b>";

            resJ48 = resJ48 + ("<b>1. Numero de instancias clasificadas:</b> "
                    + (int) evalJ48.numInstances() + "<br>");
            resJ48 = resJ48 + ("<b>2. Porcentaje de instancias correctamente "
                    + "clasificadas:</b> " + formato.format(evalJ48.pctCorrect())
                    + "<br>");
            resJ48 = resJ48 + ("<b>3. Numero de instancias correctamente "
                    + "clasificadas:</b>" + (int) evalJ48.correct() + "<br>");
            resJ48 = resJ48 + ("<b>4. Porcentaje de instancias incorrectamente "
                    + "clasificadas:</b> " + formato.format(evalJ48.pctIncorrect())
                    + "<br>");
            resJ48 = resJ48 + ("<b>5. Numero de instancias incorrectamente "
                    + "clasificadas:</b> " + (int) evalJ48.incorrect() + "<br>");
            resJ48 = resJ48 + ("<b>6. Media del error absoluto:</b> "
                    + formato.format(evalJ48.meanAbsoluteError()) + "<br>");
            resJ48 = resJ48 + ("<b>7. " + evalJ48.toMatrixString("Matriz de"
                    + " confusion</b>").replace("\n", "<br>"));

            // SE GRAFICA EL ARBOL GENERADO
            //Se crea un Jframe Temporal
            final javax.swing.JFrame jf
                    = new javax.swing.JFrame("Arbol de decision: J48");
            /*Se asigna un tamaño*/
            jf.setSize(500, 400);
            /*Se define un borde*/
            jf.getContentPane().setLayout(new BorderLayout());
            /*Se instancia la grafica del arbol, estableciendo el tipo J48
            Parametros (Listener, Tipo de arbol, Tipo de nodos)
            El placeNode2 colocará los nodos para que caigan en forma uniforme
            por debajo de su padre*/
            TreeVisualizer tv = new TreeVisualizer(null, j48.graph(),
                    new PlaceNode2());
            /*Añade el arbol centrandolo*/
            jf.getContentPane().add(tv, BorderLayout.CENTER);
            /*Añadimos un listener para la X del close*/
            jf.addWindowListener(new java.awt.event.WindowAdapter() {
                @Override
                public void windowClosing(java.awt.event.WindowEvent e) {
                    jf.dispose();
                }
            });
            /*Lo visualizamos*/
            jf.setVisible(true);
            /*Ajustamos el arbol al ancho del JFRM*/
            tv.fitToScreen();

            return resJ48;

        } catch (Exception e) {
            return "El error es" + e.getMessage();

        }
    }

    public String apriori(Instances data) {
        try {
            //Creamos un objeto de asosiacion por apriori
            Apriori aso = new Apriori();

            //creamos el descriptivo apriori con los datos
            aso.buildAssociations(data);

            /*Se cargan los resultados de loa asociacion apriori*/
            String resApriori = "<br><b><center>Resultados Asociacion "
                    + "Apriori</center><br>========<br>El modelo de asociacion "
                    + "generado indica los siguientes resultados:"
                    + "<br>========<br></b>";

            //Obtenemos resultados
            for (int i = 0; i < aso.getAssociationRules().getRules().size(); i++) {
                resApriori = resApriori + "<b>" + (i + 1) + ". Si</b> "
                        + aso.getAssociationRules().getRules().get(i).getPremise().toString();
                resApriori = resApriori + " <b>Entonces</b> "
                        + aso.getAssociationRules().getRules().get(i).getConsequence().toString();
                resApriori = resApriori + " <b>Con un</b> "
                        + (int) (aso.getAssociationRules().getRules().get(i).getPrimaryMetricValue() * 100)
                        + "% de probabilidad<br>";
            }

            return resApriori;

        } catch (Exception e) {
            return "El error es" + e.getMessage();
        }
    }

    public String mineria(String mensaje) {

        // El mensaje tipo String lo convertimos a un StringReader
        StringReader sr = new StringReader(mensaje);
        // el StringReader lo convertimos a un Buffer
        BufferedReader br = new BufferedReader(sr);

        try {
            //definimos un objeto que contendra los datos a clasificar
            Instances data = new Instances(br);
            //Seleccionamos cual sera el atributo clase                
            data.setClassIndex(data.numAttributes() - 1);
            //cerramos el objeto buffer
            br.close();

            /*Obtenemos resultados*/
            String descripcion = definirEncabezado(data);
            String resultadoBayesiano = redBayesiana(data);
            String resultadoJ48 = arbolJ48(data);
            String resultadoApriori = apriori(data);

            /*Se concatenan resultados y se envian*/
            String res = descripcion + "\n" + resultadoBayesiano + "\n" + resultadoJ48 + "\n" + resultadoApriori;//se le añade el contenido al objeto de tipo mensaje            

            return res;

        } catch (Exception e) {
            return "El error es" + e.getMessage();

        }
    }

}
