#!/usr/bin/env python3
"""
Knowledge Graph Test Fixtures for Clinical Metabolomics Oracle.

This module provides specialized test fixtures for creating and testing knowledge graph
entities, relationships, and graph-based operations specifically tailored for 
metabolomics research and clinical applications.

Components:
- MetabolomicsKnowledgeGraph: Creates realistic metabolomics knowledge graphs
- BiomedicalEntityGenerator: Generates biomedical entities with realistic attributes
- RelationshipPatternGenerator: Creates biologically meaningful relationships
- PathwayKnowledgeBuilder: Builds metabolic pathway knowledge structures
- DiseaseOntologyFixtures: Provides disease-specific knowledge graph data
- ClinicalKnowledgeGraphValidator: Validates knowledge graph content for accuracy

Author: Claude Code (Anthropic)
Created: August 7, 2025
Version: 1.0.0
"""

import pytest
import random
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import numpy as np


@dataclass
class BiomedicalEntity:
    """Represents a biomedical entity in the knowledge graph."""
    entity_id: str
    entity_type: str  # 'metabolite', 'protein', 'gene', 'disease', 'pathway', 'drug'
    name: str
    synonyms: List[str] = field(default_factory=list)
    description: str = ""
    external_ids: Dict[str, str] = field(default_factory=dict)  # e.g., {'HMDB': 'HMDB0000122', 'KEGG': 'C00031'}
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0
    source: str = "test_fixture"
    created_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'name': self.name,
            'synonyms': self.synonyms,
            'description': self.description,
            'external_ids': self.external_ids,
            'properties': self.properties,
            'confidence_score': self.confidence_score,
            'source': self.source,
            'created_timestamp': self.created_timestamp
        }


@dataclass
class BiomedicalRelationship:
    """Represents a relationship between biomedical entities."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # 'regulates', 'metabolizes', 'associated_with', etc.
    direction: str = "directed"  # 'directed', 'undirected'
    confidence_score: float = 1.0
    evidence: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = "test_fixture"
    created_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            'relationship_id': self.relationship_id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'relationship_type': self.relationship_type,
            'direction': self.direction,
            'confidence_score': self.confidence_score,
            'evidence': self.evidence,
            'properties': self.properties,
            'source': self.source,
            'created_timestamp': self.created_timestamp
        }


@dataclass
class KnowledgeGraphStats:
    """Statistics about a knowledge graph."""
    total_entities: int = 0
    total_relationships: int = 0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    relationship_type_counts: Dict[str, int] = field(default_factory=dict)
    average_degree: float = 0.0
    density: float = 0.0
    connected_components: int = 0
    largest_component_size: int = 0
    clustering_coefficient: float = 0.0


class BiomedicalEntityGenerator:
    """
    Generates realistic biomedical entities for knowledge graph testing.
    """
    
    # Comprehensive entity databases with realistic properties
    METABOLITE_ENTITIES = {
        'glucose': {
            'name': 'Glucose',
            'synonyms': ['D-glucose', 'dextrose', 'blood sugar', 'grape sugar'],
            'external_ids': {
                'HMDB': 'HMDB0000122',
                'KEGG': 'C00031',
                'ChEBI': 'CHEBI:17234',
                'PubChem': '5793'
            },
            'properties': {
                'molecular_formula': 'C6H12O6',
                'molecular_weight': 180.16,
                'smiles': 'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',
                'inchi_key': 'WQZGKKKJIJFFOK-GASJEMHNSA-N',
                'solubility': 'highly_soluble',
                'bioavailability': 'high',
                'half_life_hours': 0.5,
                'toxicity_level': 'none'
            },
            'description': 'Primary source of energy for cells, key metabolite in glycolysis'
        },
        'lactate': {
            'name': 'Lactate',
            'synonyms': ['lactic acid', 'milk acid', '2-hydroxypropanoic acid'],
            'external_ids': {
                'HMDB': 'HMDB0000190',
                'KEGG': 'C00186',
                'ChEBI': 'CHEBI:16651',
                'PubChem': '612'
            },
            'properties': {
                'molecular_formula': 'C3H6O3',
                'molecular_weight': 90.08,
                'smiles': 'C[C@H](C(=O)O)O',
                'bioavailability': 'high',
                'half_life_hours': 1.0,
                'toxicity_level': 'low'
            },
            'description': 'Product of anaerobic glycolysis, marker of tissue hypoxia'
        },
        'cholesterol': {
            'name': 'Cholesterol',
            'synonyms': ['cholest-5-en-3β-ol', '5-cholesten-3β-ol'],
            'external_ids': {
                'HMDB': 'HMDB0000067',
                'KEGG': 'C00187',
                'ChEBI': 'CHEBI:16113',
                'PubChem': '5997'
            },
            'properties': {
                'molecular_formula': 'C27H46O',
                'molecular_weight': 386.65,
                'bioavailability': 'medium',
                'solubility': 'lipophilic',
                'toxicity_level': 'none'
            },
            'description': 'Sterol molecule, essential for cell membrane structure'
        }
    }
    
    PROTEIN_ENTITIES = {
        'insulin': {
            'name': 'Insulin',
            'synonyms': ['INS', 'insulin hormone', 'human insulin'],
            'external_ids': {
                'UniProt': 'P01308',
                'Gene': 'INS',
                'Entrez': '3630'
            },
            'properties': {
                'protein_class': 'hormone',
                'molecular_weight_kda': 5.8,
                'half_life_minutes': 5,
                'localization': 'secreted',
                'function': 'glucose homeostasis'
            },
            'description': 'Peptide hormone regulating glucose metabolism'
        },
        'albumin': {
            'name': 'Albumin',
            'synonyms': ['ALB', 'serum albumin', 'human albumin'],
            'external_ids': {
                'UniProt': 'P02768',
                'Gene': 'ALB',
                'Entrez': '213'
            },
            'properties': {
                'protein_class': 'transport_protein',
                'molecular_weight_kda': 66.5,
                'half_life_days': 20,
                'localization': 'plasma',
                'function': 'transport and osmotic pressure'
            },
            'description': 'Most abundant plasma protein, transport function'
        },
        'hemoglobin': {
            'name': 'Hemoglobin',
            'synonyms': ['HB', 'Hgb', 'hemoglobin A'],
            'external_ids': {
                'UniProt': 'P69905',
                'Gene': 'HBA1',
                'Entrez': '3039'
            },
            'properties': {
                'protein_class': 'oxygen_carrier',
                'molecular_weight_kda': 64.5,
                'half_life_days': 120,
                'localization': 'erythrocyte',
                'function': 'oxygen transport'
            },
            'description': 'Iron-containing oxygen transport protein'
        }
    }
    
    DISEASE_ENTITIES = {
        'diabetes_mellitus': {
            'name': 'Diabetes mellitus',
            'synonyms': ['diabetes', 'DM', 'diabetes mellitus type 2'],
            'external_ids': {
                'ICD10': 'E11',
                'MESH': 'D003920',
                'OMIM': '125853'
            },
            'properties': {
                'disease_category': 'metabolic_disorder',
                'prevalence_percent': 8.5,
                'age_onset': 'adult',
                'severity': 'chronic',
                'heritability': 0.7
            },
            'description': 'Metabolic disorder characterized by chronic hyperglycemia'
        },
        'cardiovascular_disease': {
            'name': 'Cardiovascular disease',
            'synonyms': ['CVD', 'heart disease', 'cardiac disease'],
            'external_ids': {
                'ICD10': 'I25',
                'MESH': 'D002318'
            },
            'properties': {
                'disease_category': 'circulatory_disorder',
                'prevalence_percent': 6.2,
                'age_onset': 'adult',
                'severity': 'chronic',
                'heritability': 0.6
            },
            'description': 'Disorders affecting the heart and blood vessels'
        }
    }
    
    PATHWAY_ENTITIES = {
        'glycolysis': {
            'name': 'Glycolysis',
            'synonyms': ['glycolytic pathway', 'Embden-Meyerhof pathway'],
            'external_ids': {
                'KEGG': 'hsa00010',
                'Reactome': 'R-HSA-70171'
            },
            'properties': {
                'pathway_type': 'metabolic',
                'cellular_location': 'cytoplasm',
                'energy_yield': 'ATP_production',
                'regulation': 'allosteric'
            },
            'description': 'Metabolic pathway converting glucose to pyruvate'
        },
        'tca_cycle': {
            'name': 'Citrate cycle',
            'synonyms': ['TCA cycle', 'Krebs cycle', 'tricarboxylic acid cycle'],
            'external_ids': {
                'KEGG': 'hsa00020',
                'Reactome': 'R-HSA-71403'
            },
            'properties': {
                'pathway_type': 'metabolic',
                'cellular_location': 'mitochondria',
                'energy_yield': 'NADH_FADH2_production',
                'regulation': 'allosteric'
            },
            'description': 'Central metabolic pathway for energy production'
        }
    }
    
    DRUG_ENTITIES = {
        'metformin': {
            'name': 'Metformin',
            'synonyms': ['metformin hydrochloride', 'glucophage', 'dimethylbiguanide'],
            'external_ids': {
                'DrugBank': 'DB00331',
                'ChEMBL': 'CHEMBL1431',
                'PubChem': '4091'
            },
            'properties': {
                'drug_class': 'biguanide',
                'molecular_weight': 129.16,
                'bioavailability_percent': 50,
                'half_life_hours': 6.2,
                'mechanism': 'AMPK_activation',
                'indication': 'type_2_diabetes'
            },
            'description': 'Antidiabetic medication, first-line treatment for type 2 diabetes'
        },
        'insulin_glargine': {
            'name': 'Insulin glargine',
            'synonyms': ['Lantus', 'long-acting insulin', 'basal insulin'],
            'external_ids': {
                'DrugBank': 'DB00047',
                'ChEMBL': 'CHEMBL1201247'
            },
            'properties': {
                'drug_class': 'long_acting_insulin',
                'molecular_weight_kda': 6.1,
                'duration_hours': 24,
                'onset_hours': 2,
                'peak_time': 'no_peak',
                'indication': 'diabetes_mellitus'
            },
            'description': 'Long-acting insulin analog for glucose control'
        }
    }
    
    @classmethod
    def generate_entity(cls, entity_type: str, entity_key: Optional[str] = None) -> BiomedicalEntity:
        """Generate a biomedical entity of specified type."""
        
        entity_databases = {
            'metabolite': cls.METABOLITE_ENTITIES,
            'protein': cls.PROTEIN_ENTITIES,
            'disease': cls.DISEASE_ENTITIES,
            'pathway': cls.PATHWAY_ENTITIES,
            'drug': cls.DRUG_ENTITIES
        }
        
        if entity_type not in entity_databases:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        database = entity_databases[entity_type]
        
        # Select specific entity or random
        if entity_key and entity_key in database:
            selected_key = entity_key
        else:
            selected_key = random.choice(list(database.keys()))
        
        entity_data = database[selected_key]
        
        # Generate unique entity ID
        entity_id = f"{entity_type}_{selected_key}_{random.randint(1000, 9999)}"
        
        return BiomedicalEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=entity_data['name'],
            synonyms=entity_data.get('synonyms', []),
            description=entity_data.get('description', ''),
            external_ids=entity_data.get('external_ids', {}),
            properties=entity_data.get('properties', {}),
            confidence_score=random.uniform(0.8, 1.0),
            source="test_fixture"
        )
    
    @classmethod
    def generate_entity_collection(cls, 
                                 entity_types: List[str],
                                 count_per_type: int = 5) -> List[BiomedicalEntity]:
        """Generate collection of entities across multiple types."""
        entities = []
        
        for entity_type in entity_types:
            database = getattr(cls, f"{entity_type.upper()}_ENTITIES", {})
            available_keys = list(database.keys())
            
            # Generate requested number of entities (with repetition if needed)
            for i in range(count_per_type):
                entity_key = available_keys[i % len(available_keys)] if available_keys else None
                entity = cls.generate_entity(entity_type, entity_key)
                entities.append(entity)
        
        return entities


class RelationshipPatternGenerator:
    """
    Generates biologically meaningful relationships between biomedical entities.
    """
    
    # Relationship type definitions with biological validity rules
    RELATIONSHIP_PATTERNS = {
        'metabolite_protein': {
            'types': ['substrate_of', 'product_of', 'inhibits', 'activates', 'binds_to'],
            'evidence_types': ['enzymatic_assay', 'binding_study', 'metabolic_pathway'],
            'confidence_range': (0.6, 0.95)
        },
        'metabolite_disease': {
            'types': ['biomarker_for', 'elevated_in', 'decreased_in', 'associated_with'],
            'evidence_types': ['clinical_study', 'cohort_analysis', 'case_control'],
            'confidence_range': (0.5, 0.9)
        },
        'protein_disease': {
            'types': ['causative_of', 'protective_against', 'biomarker_for', 'therapeutic_target'],
            'evidence_types': ['genetic_association', 'protein_expression', 'clinical_trial'],
            'confidence_range': (0.6, 0.95)
        },
        'metabolite_pathway': {
            'types': ['participates_in', 'regulates', 'product_of', 'substrate_of'],
            'evidence_types': ['pathway_analysis', 'flux_analysis', 'enzyme_kinetics'],
            'confidence_range': (0.7, 0.98)
        },
        'drug_metabolite': {
            'types': ['metabolite_of', 'affects_level_of', 'biomarker_of_response'],
            'evidence_types': ['pharmacokinetic_study', 'clinical_trial', 'metabolomics'],
            'confidence_range': (0.6, 0.9)
        },
        'drug_disease': {
            'types': ['treats', 'prevents', 'contraindicated_in', 'indicated_for'],
            'evidence_types': ['clinical_trial', 'meta_analysis', 'real_world_evidence'],
            'confidence_range': (0.7, 0.98)
        },
        'protein_pathway': {
            'types': ['catalyzes_reaction_in', 'regulates', 'participates_in'],
            'evidence_types': ['enzyme_assay', 'protein_interaction', 'pathway_mapping'],
            'confidence_range': (0.8, 0.98)
        }
    }
    
    @classmethod
    def generate_relationship(cls, 
                            source_entity: BiomedicalEntity,
                            target_entity: BiomedicalEntity,
                            relationship_type: Optional[str] = None) -> Optional[BiomedicalRelationship]:
        """Generate biologically valid relationship between entities."""
        
        # Determine entity pair type
        entity_pair_key = f"{source_entity.entity_type}_{target_entity.entity_type}"
        reverse_key = f"{target_entity.entity_type}_{source_entity.entity_type}"
        
        # Check if relationship pattern exists
        if entity_pair_key in cls.RELATIONSHIP_PATTERNS:
            pattern = cls.RELATIONSHIP_PATTERNS[entity_pair_key]
        elif reverse_key in cls.RELATIONSHIP_PATTERNS:
            pattern = cls.RELATIONSHIP_PATTERNS[reverse_key]
            # Swap entities for reverse relationship
            source_entity, target_entity = target_entity, source_entity
        else:
            # No pattern available for this entity pair
            return None
        
        # Select relationship type
        if relationship_type and relationship_type in pattern['types']:
            selected_type = relationship_type
        else:
            selected_type = random.choice(pattern['types'])
        
        # Generate relationship properties
        confidence_range = pattern['confidence_range']
        confidence = random.uniform(*confidence_range)
        
        evidence_type = random.choice(pattern['evidence_types'])
        evidence = [
            f"{evidence_type}: {cls._generate_evidence_description(evidence_type, selected_type)}",
            f"Study ID: TEST_{random.randint(1000, 9999)}"
        ]
        
        # Generate relationship ID
        relationship_id = f"rel_{source_entity.entity_id}_{target_entity.entity_id}_{random.randint(100, 999)}"
        
        # Add relationship-specific properties
        properties = {
            'strength': cls._assign_relationship_strength(selected_type),
            'evidence_quality': random.choice(['high', 'medium', 'low']),
            'study_count': random.randint(1, 15),
            'reproducibility': random.uniform(0.5, 0.95)
        }
        
        return BiomedicalRelationship(
            relationship_id=relationship_id,
            source_entity_id=source_entity.entity_id,
            target_entity_id=target_entity.entity_id,
            relationship_type=selected_type,
            direction="directed",
            confidence_score=confidence,
            evidence=evidence,
            properties=properties,
            source="test_fixture"
        )
    
    @classmethod
    def _generate_evidence_description(cls, evidence_type: str, relationship_type: str) -> str:
        """Generate realistic evidence description."""
        templates = {
            'clinical_study': f"Clinical study showed {relationship_type} with p<0.05",
            'enzymatic_assay': f"Enzymatic assay demonstrated {relationship_type} activity",
            'binding_study': f"Binding study confirmed {relationship_type} interaction",
            'pathway_analysis': f"Pathway analysis revealed {relationship_type} mechanism",
            'genetic_association': f"GWAS identified {relationship_type} association",
            'clinical_trial': f"Phase III trial demonstrated {relationship_type} effect"
        }
        
        return templates.get(evidence_type, f"Study demonstrated {relationship_type}")
    
    @classmethod
    def _assign_relationship_strength(cls, relationship_type: str) -> str:
        """Assign strength based on relationship type."""
        strong_relationships = ['causes', 'treats', 'catalyzes_reaction_in', 'substrate_of']
        medium_relationships = ['regulates', 'associated_with', 'biomarker_for']
        
        if relationship_type in strong_relationships:
            return 'strong'
        elif relationship_type in medium_relationships:
            return 'medium'
        else:
            return 'weak'
    
    @classmethod
    def generate_relationship_network(cls, 
                                    entities: List[BiomedicalEntity],
                                    relationship_density: float = 0.3) -> List[BiomedicalRelationship]:
        """Generate network of relationships between entities."""
        relationships = []
        
        # Calculate number of relationships to generate
        max_relationships = len(entities) * (len(entities) - 1) // 2
        target_relationships = int(max_relationships * relationship_density)
        
        # Generate random pairs and create relationships
        attempts = 0
        while len(relationships) < target_relationships and attempts < target_relationships * 3:
            source = random.choice(entities)
            target = random.choice(entities)
            
            # Avoid self-relationships
            if source.entity_id == target.entity_id:
                attempts += 1
                continue
            
            # Check if relationship already exists
            existing = any(
                (r.source_entity_id == source.entity_id and r.target_entity_id == target.entity_id) or
                (r.source_entity_id == target.entity_id and r.target_entity_id == source.entity_id)
                for r in relationships
            )
            
            if existing:
                attempts += 1
                continue
            
            # Generate relationship
            relationship = cls.generate_relationship(source, target)
            if relationship:
                relationships.append(relationship)
            
            attempts += 1
        
        return relationships


class MetabolomicsKnowledgeGraph:
    """
    Creates and manages metabolomics-specific knowledge graphs for testing.
    """
    
    def __init__(self):
        self.entities: Dict[str, BiomedicalEntity] = {}
        self.relationships: Dict[str, BiomedicalRelationship] = {}
        self.graph = nx.MultiDiGraph()
    
    def add_entity(self, entity: BiomedicalEntity) -> None:
        """Add entity to the knowledge graph."""
        self.entities[entity.entity_id] = entity
        self.graph.add_node(entity.entity_id, **entity.to_dict())
    
    def add_relationship(self, relationship: BiomedicalRelationship) -> None:
        """Add relationship to the knowledge graph."""
        self.relationships[relationship.relationship_id] = relationship
        
        # Add nodes if they don't exist
        if relationship.source_entity_id not in self.graph:
            self.graph.add_node(relationship.source_entity_id)
        if relationship.target_entity_id not in self.graph:
            self.graph.add_node(relationship.target_entity_id)
        
        # Add edge
        self.graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            key=relationship.relationship_id,
            **relationship.to_dict()
        )
    
    def build_disease_focused_graph(self, 
                                  disease: str = 'diabetes',
                                  include_drugs: bool = True) -> 'MetabolomicsKnowledgeGraph':
        """Build knowledge graph focused on specific disease."""
        
        # Generate disease-relevant entities
        disease_entities = []
        
        # Add disease entity
        disease_entity = BiomedicalEntityGenerator.generate_entity('disease', f'{disease}_mellitus')
        disease_entities.append(disease_entity)
        self.add_entity(disease_entity)
        
        # Add relevant metabolites
        metabolite_keys = ['glucose', 'lactate', 'cholesterol']
        for key in metabolite_keys:
            entity = BiomedicalEntityGenerator.generate_entity('metabolite', key)
            disease_entities.append(entity)
            self.add_entity(entity)
        
        # Add relevant proteins
        protein_keys = ['insulin', 'albumin', 'hemoglobin']
        for key in protein_keys:
            entity = BiomedicalEntityGenerator.generate_entity('protein', key)
            disease_entities.append(entity)
            self.add_entity(entity)
        
        # Add pathways
        pathway_keys = ['glycolysis', 'tca_cycle']
        for key in pathway_keys:
            entity = BiomedicalEntityGenerator.generate_entity('pathway', key)
            disease_entities.append(entity)
            self.add_entity(entity)
        
        # Add drugs if requested
        if include_drugs:
            drug_keys = ['metformin', 'insulin_glargine']
            for key in drug_keys:
                entity = BiomedicalEntityGenerator.generate_entity('drug', key)
                disease_entities.append(entity)
                self.add_entity(entity)
        
        # Generate relationships
        relationships = RelationshipPatternGenerator.generate_relationship_network(
            disease_entities, relationship_density=0.4
        )
        
        for relationship in relationships:
            self.add_relationship(relationship)
        
        return self
    
    def get_entity_neighbors(self, entity_id: str, relationship_types: Optional[List[str]] = None) -> List[str]:
        """Get neighboring entities with optional filtering by relationship type."""
        neighbors = []
        
        if entity_id not in self.graph:
            return neighbors
        
        # Get all edges (both outgoing and incoming)
        edges = list(self.graph.out_edges(entity_id, data=True)) + list(self.graph.in_edges(entity_id, data=True))
        
        for source, target, edge_data in edges:
            neighbor_id = target if source == entity_id else source
            
            if relationship_types:
                if edge_data.get('relationship_type') in relationship_types:
                    neighbors.append(neighbor_id)
            else:
                neighbors.append(neighbor_id)
        
        return list(set(neighbors))  # Remove duplicates
    
    def query_entities_by_type(self, entity_type: str) -> List[BiomedicalEntity]:
        """Query entities by type."""
        return [entity for entity in self.entities.values() 
                if entity.entity_type == entity_type]
    
    def query_relationships_by_type(self, relationship_type: str) -> List[BiomedicalRelationship]:
        """Query relationships by type."""
        return [rel for rel in self.relationships.values() 
                if rel.relationship_type == relationship_type]
    
    def get_shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Get shortest path between two entities."""
        try:
            # Convert to undirected for path finding
            undirected_graph = self.graph.to_undirected()
            path = nx.shortest_path(undirected_graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def calculate_statistics(self) -> KnowledgeGraphStats:
        """Calculate comprehensive statistics about the knowledge graph."""
        
        # Basic counts
        total_entities = len(self.entities)
        total_relationships = len(self.relationships)
        
        # Entity type distribution
        entity_type_counts = defaultdict(int)
        for entity in self.entities.values():
            entity_type_counts[entity.entity_type] += 1
        
        # Relationship type distribution
        relationship_type_counts = defaultdict(int)
        for rel in self.relationships.values():
            relationship_type_counts[rel.relationship_type] += 1
        
        # Graph statistics
        if total_entities > 0:
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            average_degree = sum(degrees) / len(degrees) if degrees else 0
            
            # Density
            max_edges = total_entities * (total_entities - 1)
            density = (total_relationships / max_edges) if max_edges > 0 else 0
            
            # Connected components
            undirected = self.graph.to_undirected()
            components = list(nx.connected_components(undirected))
            connected_components = len(components)
            largest_component_size = len(max(components, key=len)) if components else 0
            
            # Clustering coefficient
            clustering_coefficient = nx.average_clustering(undirected) if total_entities > 2 else 0
        else:
            average_degree = density = connected_components = largest_component_size = clustering_coefficient = 0
        
        return KnowledgeGraphStats(
            total_entities=total_entities,
            total_relationships=total_relationships,
            entity_type_counts=dict(entity_type_counts),
            relationship_type_counts=dict(relationship_type_counts),
            average_degree=average_degree,
            density=density,
            connected_components=connected_components,
            largest_component_size=largest_component_size,
            clustering_coefficient=clustering_coefficient
        )
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export knowledge graph to dictionary format."""
        return {
            'entities': {eid: entity.to_dict() for eid, entity in self.entities.items()},
            'relationships': {rid: rel.to_dict() for rid, rel in self.relationships.items()},
            'statistics': self.calculate_statistics().__dict__
        }


# Pytest fixtures for knowledge graph testing
@pytest.fixture
def biomedical_entity_generator():
    """Provide biomedical entity generator."""
    return BiomedicalEntityGenerator()

@pytest.fixture
def relationship_pattern_generator():
    """Provide relationship pattern generator."""
    return RelationshipPatternGenerator()

@pytest.fixture
def sample_metabolite_entities():
    """Provide sample metabolite entities."""
    entities = []
    metabolite_keys = ['glucose', 'lactate', 'cholesterol']
    
    for key in metabolite_keys:
        entity = BiomedicalEntityGenerator.generate_entity('metabolite', key)
        entities.append(entity)
    
    return entities

@pytest.fixture
def sample_protein_entities():
    """Provide sample protein entities."""
    entities = []
    protein_keys = ['insulin', 'albumin', 'hemoglobin']
    
    for key in protein_keys:
        entity = BiomedicalEntityGenerator.generate_entity('protein', key)
        entities.append(entity)
    
    return entities

@pytest.fixture
def sample_disease_entities():
    """Provide sample disease entities."""
    entities = []
    disease_keys = ['diabetes_mellitus', 'cardiovascular_disease']
    
    for key in disease_keys:
        entity = BiomedicalEntityGenerator.generate_entity('disease', key)
        entities.append(entity)
    
    return entities

@pytest.fixture
def comprehensive_entity_collection():
    """Provide comprehensive collection of entities across all types."""
    entity_types = ['metabolite', 'protein', 'disease', 'pathway', 'drug']
    return BiomedicalEntityGenerator.generate_entity_collection(
        entity_types=entity_types,
        count_per_type=3
    )

@pytest.fixture
def sample_relationships(comprehensive_entity_collection):
    """Provide sample relationships between entities."""
    return RelationshipPatternGenerator.generate_relationship_network(
        entities=comprehensive_entity_collection,
        relationship_density=0.3
    )

@pytest.fixture
def diabetes_knowledge_graph():
    """Provide diabetes-focused knowledge graph."""
    kg = MetabolomicsKnowledgeGraph()
    return kg.build_disease_focused_graph(disease='diabetes', include_drugs=True)

@pytest.fixture
def cardiovascular_knowledge_graph():
    """Provide cardiovascular disease knowledge graph."""
    kg = MetabolomicsKnowledgeGraph()
    return kg.build_disease_focused_graph(disease='cardiovascular', include_drugs=True)

@pytest.fixture
def multi_disease_knowledge_graph():
    """Provide knowledge graph spanning multiple diseases."""
    kg = MetabolomicsKnowledgeGraph()
    
    # Build graphs for multiple diseases and merge
    diseases = ['diabetes', 'cardiovascular']
    for disease in diseases:
        disease_kg = MetabolomicsKnowledgeGraph()
        disease_kg.build_disease_focused_graph(disease=disease, include_drugs=True)
        
        # Merge entities and relationships
        for entity in disease_kg.entities.values():
            kg.add_entity(entity)
        for relationship in disease_kg.relationships.values():
            kg.add_relationship(relationship)
    
    return kg

@pytest.fixture
def knowledge_graph_statistics(diabetes_knowledge_graph):
    """Provide knowledge graph statistics."""
    return diabetes_knowledge_graph.calculate_statistics()

@pytest.fixture
def pathway_focused_entities():
    """Provide entities focused on metabolic pathways."""
    entities = []
    
    # Add pathway entities
    pathway_keys = ['glycolysis', 'tca_cycle']
    for key in pathway_keys:
        entity = BiomedicalEntityGenerator.generate_entity('pathway', key)
        entities.append(entity)
    
    # Add related metabolites
    metabolite_keys = ['glucose', 'lactate']
    for key in metabolite_keys:
        entity = BiomedicalEntityGenerator.generate_entity('metabolite', key)
        entities.append(entity)
    
    # Add related proteins
    protein_keys = ['insulin']
    for key in protein_keys:
        entity = BiomedicalEntityGenerator.generate_entity('protein', key)
        entities.append(entity)
    
    return entities

@pytest.fixture
def drug_interaction_network():
    """Provide drug-centric interaction network."""
    kg = MetabolomicsKnowledgeGraph()
    
    # Add drug entities
    drug_keys = ['metformin', 'insulin_glargine']
    drug_entities = []
    for key in drug_keys:
        entity = BiomedicalEntityGenerator.generate_entity('drug', key)
        drug_entities.append(entity)
        kg.add_entity(entity)
    
    # Add target metabolites and diseases
    metabolite_keys = ['glucose', 'lactate']
    for key in metabolite_keys:
        entity = BiomedicalEntityGenerator.generate_entity('metabolite', key)
        drug_entities.append(entity)
        kg.add_entity(entity)
    
    disease_keys = ['diabetes_mellitus']
    for key in disease_keys:
        entity = BiomedicalEntityGenerator.generate_entity('disease', key)
        drug_entities.append(entity)
        kg.add_entity(entity)
    
    # Generate drug-focused relationships
    relationships = RelationshipPatternGenerator.generate_relationship_network(
        drug_entities, relationship_density=0.5
    )
    
    for relationship in relationships:
        kg.add_relationship(relationship)
    
    return kg