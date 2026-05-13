# First-Sentence Comparison: Fine-Tuned vs Base

This shows how each model's **first response** compares to the textbook answer.

---

## Q1: Subject: Chemistry
Question: Why is LPG preferred as domestic fuel?
**Subject:** Chemistry

**📖 Textbook:** High calorific value and clean burning.

**🧪 Fine-Tuned (1st sentence):** It has higher energy density and burns cleanly.

**🔹 Base Model (1st sentence):** In the context of domestic use, LPG (Liquefied Petroleum Gas) is preferred over other fuels for several reasons:

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.2857 | 0.0000 |
| BERTScore | 0.9066 | 0.8396 |
| Cosine Sim | 0.5371 | 0.1425 |
| Directness | 1.00 | 0.70 |
| Conciseness | 0.2577 | 0.2012 |

---

## Q2: Subject: Chemistry
Question: Why is benzene more stable than expected according to Kekulé’s structure?
**Subject:** Chemistry

**📖 Textbook:** Delocalization of π-electrons over six carbon atoms leads to resonance stabilization.

**🧪 Fine-Tuned (1st sentence):** Because of resonance stabilization.

**🔹 Base Model (1st sentence):** To answer your question, let's first discuss Kekulé's structure of benzene.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.4000 | 0.0833 |
| BERTScore | 0.8923 | 0.8385 |
| Cosine Sim | 0.5175 | 0.2903 |
| Directness | 1.00 | 0.40 |
| Conciseness | 0.3491 | 0.2619 |

---

## Q3: Subject: Chemistry
Question: Why does CH₃CH₂Br react faster in ethanol than in water?
**Subject:** Chemistry

**📖 Textbook:** Ethanol favors SN2 mechanism; water favors elimination.

**🧪 Fine-Tuned (1st sentence):** Ethanol acts as polar solvent facilitating SN2 substitution; water is less effective.

**🔹 Base Model (1st sentence):** I'd be happy to help answer your question!

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.3158 | 0.0000 |
| BERTScore | 0.9245 | 0.8086 |
| Cosine Sim | 0.9011 | 0.0520 |
| Directness | 1.00 | 0.70 |
| Conciseness | 0.3727 | 0.2526 |

---

## Q4: Subject: Chemistry
Question: Explain why Al³⁺ ions are acidic in aqueous solution.
**Subject:** Chemistry

**📖 Textbook:** Al³⁺ polarizes water → H⁺ released → acidic solution.

**🧪 Fine-Tuned (1st sentence):** Al³⁺ ions can accept a proton, forming Al³⁺H⁺ complex, making the solution acidic.

**🔹 Base Model (1st sentence):** Aluminum ion, Al³⁺, is acidic in aqueous solution due to its tendency to gain hydration water molecules through a process called hydrolysis. This process forms hydroxide ions (OH⁻) and aluminum hydroxide species. The release of hydroxide ions in water makes the solution acidic.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.2857 | 0.1600 |
| BERTScore | 0.8916 | 0.8555 |
| Cosine Sim | 0.8102 | 0.7260 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.4135 | 0.3716 |

---

## Q5: Subject: Chemistry
Question: Why is CH₃CH₂OH a better solvent than hexane for polar compounds?
**Subject:** Chemistry

**📖 Textbook:** Ethanol is polar and can form hydrogen bonds with solute; hexane is nonpolar.

**🧪 Fine-Tuned (1st sentence):** –OH group provides hydrogen bonding and dipole–dipole interactions.

**🔹 Base Model (1st sentence):** I'm glad you've asked this question!

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.1818 | 0.0000 |
| BERTScore | 0.8551 | 0.7964 |
| Cosine Sim | 0.3646 | -0.0625 |
| Directness | 1.00 | 0.60 |
| Conciseness | 0.4535 | 0.3875 |

---

## Q6: Subject: Chemistry
Question: Why are gases more compressible than liquids?
**Subject:** Chemistry

**📖 Textbook:** Because gas molecules are far apart and can be pushed closer together.

**🧪 Fine-Tuned (1st sentence):** Because gases have large intermolecular distances and can be easily compressed.

**🔹 Base Model (1st sentence):** In a gas, the molecules are much farther apart from each other compared to a liquid.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.3478 | 0.2857 |
| BERTScore | 0.9130 | 0.8927 |
| Cosine Sim | 0.8319 | 0.7589 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.2868 | 0.4527 |

---

## Q7: Subject: Physics
Question: Define gamma ray.
**Subject:** Physics

**📖 Textbook:** High-energy electromagnetic radiation emitted by a nucleus.

**🧪 Fine-Tuned (1st sentence):** High-energy electromagnetic radiation produced in nuclear reactions.

**🔹 Base Model (1st sentence):** Gamma rays are the highest energy form of electromagnetic radiation.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.5000 | 0.3333 |
| BERTScore | 0.9458 | 0.8982 |
| Cosine Sim | 0.8316 | 0.6917 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.3750 | 0.3733 |

---

## Q8: Subject: Physics
Question: Define gravitational potential energy.
**Subject:** Physics

**📖 Textbook:** Energy possessed by a body due to its position in gravitational field.

**🧪 Fine-Tuned (1st sentence):** Energy possessed by a body due to its position in a gravitational field.

**🔹 Base Model (1st sentence):** In physics, the gravitational potential energy (GPE) of an object is the energy an object possesses due to its position within a gravitational field.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.9600 | 0.4444 |
| BERTScore | 0.9912 | 0.8834 |
| Cosine Sim | 0.9968 | 0.6499 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.4451 | 0.4566 |

---

## Q9: Subject: Physics
Question: Why is gravitational force conservative?
**Subject:** Physics

**📖 Textbook:** Work done by gravity depends only on initial and final positions.

**🧪 Fine-Tuned (1st sentence):** Because it depends only on position and not on the history of motion.

**🔹 Base Model (1st sentence):** I'm glad you're asking about the conservation of gravitational force in the context of physics. Gravitational force is conservative because it is a potential energy field.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.3333 | 0.0000 |
| BERTScore | 0.9008 | 0.8458 |
| Cosine Sim | 0.3485 | 0.2656 |
| Directness | 1.00 | 0.60 |
| Conciseness | 0.4273 | 0.2680 |

---

## Q10: Subject: Physics
Question: Why do sound waves reflect to produce echoes?
**Subject:** Physics

**📖 Textbook:** Because sound waves undergo reflection at surfaces.

**🧪 Fine-Tuned (1st sentence):** Because sound is a longitudinal wave that travels in a medium.

**🔹 Base Model (1st sentence):** In a closed or reflective environment, sound waves do not disappear when they encounter a surface, instead, they bounce back or reflect.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.3333 | 0.2069 |
| BERTScore | 0.9016 | 0.9065 |
| Cosine Sim | 0.7804 | 0.7167 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.2417 | 0.3854 |

---

## Q11: Subject: Physics
Question: Define strain.
**Subject:** Physics

**📖 Textbook:** Ratio of change in dimension to original dimension.

**🧪 Fine-Tuned (1st sentence):** Partial deformation per unit length.

**🔹 Base Model (1st sentence):** Strain in the context of physics, particularly in the field of mechanics, refers to the partial deformation or displacement of an object under an applied load or force.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.0000 | 0.1667 |
| BERTScore | 0.8743 | 0.8350 |
| Cosine Sim | 0.4029 | 0.1272 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.2855 | 0.3212 |

---

## Q12: Subject: Physics
Question: Why are ICs preferred over discrete components?
**Subject:** Physics

**📖 Textbook:** Because they are compact, efficient, and reliable.

**🧪 Fine-Tuned (1st sentence):** Because ICs integrate multiple components into a single chip.

**🔹 Base Model (1st sentence):** In modern electronic systems, Integrated Circuits (ICs) are preferred over discrete components for several reasons:

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.1250 | 0.0909 |
| BERTScore | 0.8730 | 0.8530 |
| Cosine Sim | 0.4745 | 0.3868 |
| Directness | 1.00 | 1.00 |
| Conciseness | 0.2974 | 0.2218 |

---

## Q13: Subject: Computer Science
Question: Explain the difference between procedural and object-oriented programming.
**Subject:** Computer Science

**📖 Textbook:** Procedural: code structured with procedures/functions. OOP: code structured with classes/objects, supports encapsulation, inheritance, polymorphism.

**🧪 Fine-Tuned (1st sentence):** Procedural: focuses on functions, data is global. OOP: uses classes, objects, encapsulation, inheritance.

**🔹 Base Model (1st sentence):** I'd be happy to help explain the difference between procedural and object-oriented programming!

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.4828 | 0.1290 |
| BERTScore | 0.9238 | 0.8419 |
| Cosine Sim | 0.8908 | 0.7546 |
| Directness | 1.00 | 0.60 |
| Conciseness | 0.5000 | 0.3309 |

---

## Q14: Subject: Computer Science
Question: Explain what a deadlock is and one way to prevent it.
**Subject:** Computer Science

**📖 Textbook:** Deadlock occurs when processes wait indefinitely for resources. Prevention: avoid circular wait by resource allocation policies.

**🧪 Fine-Tuned (1st sentence):** Deadlock occurs when two processes wait for each other indefinitely; one way to prevent it is by using resource allocation in a predetermined order.

**🔹 Base Model (1st sentence):** A deadlock is a situation that can occur in a concurrent system when two or more processes are each waiting for the other to release a resource or take a specific action. This results in both processes being unable to proceed, leading to a standstill or a state of mutual exclusion.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.5000 | 0.2090 |
| BERTScore | 0.9260 | 0.8650 |
| Cosine Sim | 0.9174 | 0.8167 |
| Directness | 1.00 | 0.80 |
| Conciseness | 0.4522 | 0.3165 |

---

## Q15: Subject: Computer Science
Question: Explain the difference between primary key and foreign key in databases.
**Subject:** Computer Science

**📖 Textbook:** Primary key uniquely identifies a record in a table; foreign key refers to a primary key in another table to enforce referential integrity.

**🧪 Fine-Tuned (1st sentence):** Primary key uniquely identifies a record in a table; foreign key establishes a relationship with another table's primary key.

**🔹 Base Model (1st sentence):** Absolutely, I'd be happy to help explain the difference between a primary key and a foreign key in the context of databases.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.6512 | 0.2609 |
| BERTScore | 0.9494 | 0.8613 |
| Cosine Sim | 0.9319 | 0.7334 |
| Directness | 1.00 | 0.60 |
| Conciseness | 0.5300 | 0.4037 |

---

## Q16: Subject: Computer Science
Question: Explain the difference between IPv4 and IPv6 in terms of packet headers and address space.
**Subject:** Computer Science

**📖 Textbook:** IPv4: 32-bit address, simpler headers, limited addresses. IPv6: 128-bit address, more complex headers, huge address space.

**🧪 Fine-Tuned (1st sentence):** IPv4: 20-byte header, 32-bit address space (4 billion addresses). IPv6: 40-byte header, 128-bit address space (340 undecillion addresses).

**🔹 Base Model (1st sentence):** I'd be happy to help explain the differences between IPv4 and IPv6, particularly in the context of packet headers and address space.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.5000 | 0.2439 |
| BERTScore | 0.9151 | 0.8744 |
| Cosine Sim | 0.9067 | 0.7672 |
| Directness | 1.00 | 0.70 |
| Conciseness | 0.4518 | 0.3240 |

---

## Q17: Subject: Computer Science
Question: What is a hash function and where is it used?
**Subject:** Computer Science

**📖 Textbook:** A hash function maps input data to a fixed-size value (hash). Used in hash tables, encryption, and data integrity.

**🧪 Fine-Tuned (1st sentence):** A hash function is used to map a large data set to a smaller fixed size. It is used in various applications like database indexing, password storage, and in data structures like hash tables.

**🔹 Base Model (1st sentence):** A hash function is a mathematical function that maps data of an arbitrary size to a fixed size. The result is called a hash value or simply a hash. It's a quick way to compute a unique representation of an input, which is particularly useful for indexing and retrieving data in a database.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.4815 | 0.3784 |
| BERTScore | 0.9161 | 0.9006 |
| Cosine Sim | 0.8562 | 0.8317 |
| Directness | 1.00 | 0.90 |
| Conciseness | 0.4224 | 0.3975 |

---

## Q18: Subject: Computer Science
Question: Explain the difference between compiler and interpreter.
**Subject:** Computer Science

**📖 Textbook:** Compiler translates the whole program to machine code before execution; interpreter translates and executes code line by line.

**🧪 Fine-Tuned (1st sentence):** Compiler translates entire program before execution; interpreter translates line by line during execution.

**🔹 Base Model (1st sentence):** In the context of programming, both compilers and interpreters are tools used to execute code. However, they operate in different ways.

**Winner:** ✅ Fine-Tuned

| Metric | Fine-Tuned | Base |
|--------|:---------:|:----:|
| ROUGE-L | 0.6452 | 0.2564 |
| BERTScore | 0.9529 | 0.8640 |
| Cosine Sim | 0.9220 | 0.7222 |
| Directness | 1.00 | 0.70 |
| Conciseness | 0.4507 | 0.3644 |

---
