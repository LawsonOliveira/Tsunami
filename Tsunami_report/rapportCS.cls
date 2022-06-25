\LoadClass[12pt,a4paper]{report} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage[english]{babel} %Langue du document
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage[T1]{fontenc} %Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=1.5cm,right=1.5cm,top=1cm,bottom=3.1cm]{geometry} %Configuration de la page

\RequirePackage{titlesec}
\titleformat{\chapter}
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter \ -}{5pt}{\Huge}   
\titlespacing*{\chapter}{0pt}{-30pt}{5pt}
\titlespacing{\section}{0pt}{2.5ex}{0ex}
\titlespacing{\subsection}{0pt}{1ex}{0ex}
\titlespacing{\subsubsection}{0pt}{0.5ex}{0ex}
\titleformat{\section}
  {\normalfont\fontsize{18}{18}\bfseries}{\thesection}{1em}{}    
%-------------------- Informations sur le rapport ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
%\fancyhead[LE,RO]{foo}
%\fancyhead[LO,RE]{bar}

\pagestyle{fancyplain}
\renewcommand\plainheadrulewidth{.4pt}% headrule on plain pages
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.05]{logos/logo_ECS.png}} %Affichage de l'image au top de la page
\rhead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{\textbf{Tsunami propagation using neural networks}}
\lfoot{\UE}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}

\ThisLRCornerWallPaper{0.3}{logos/logoECS.png}
	\centering %Centraliser le contenu
	\includegraphics[width=0.75\textwidth]{logos/logo.jpg}\par\vspace{1cm} %Insertion du logo
	{\scshape\LARGE CentraleSupélec \par} %Nom de l'université
	\vspace{1.5cm}%Espace de 1,5cm
	{\scshape\Large \UE \\ \sujet \\ Pôle projet 05 - Formation à la recherce\par} %sous-titre
	\vspace{0.5cm}%Espace de 1cm
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\huge\bfseries \titre \par} \
    \rule{\linewidth}{0.2 mm} \\[1.5 cm]
	\vspace{0.5cm}%Espace de 3cm
    
	\begin{minipage}{0.5\textwidth} %Minipage pour faire deux colonnes
		\begin{flushleft} \large %Envoyer à gauche
		\emph{\textbf{Group:}}\\ %Pour le titre au dessus des noms à gauche
        \eleves\\ %Remplacer pour chacun
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Professors:}} \\
		 \enseignant \\
		\end{flushright}
	\end{minipage}\\[4cm]
    
	\vfill
	{\large \today\par} %Affichage de la date

\end{titlepage}
}
\newcommand{\tabledematieres}{
\tableofcontents
\thispagestyle{empty}
\setcounter{page}{0}
\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}



